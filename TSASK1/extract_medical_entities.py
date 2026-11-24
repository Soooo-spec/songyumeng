import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_BASE_URL = "https://ws-ad5zonuzcotnpkmh.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def load_env_file(env_path: Path | None = None) -> None:
    """自动加载 .env 文件中的环境变量（不覆盖已有系统环境变量）。"""
    if env_path is not None:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            raise FileNotFoundError(f"指定的环境配置文件不存在: {env_path}")
    else:
        load_dotenv(override=False)


def extract_pdf_text(pdf_path: Path) -> str:
    """使用 PyMuPDF 提取 PDF 全文文本（按页标注页码）。"""
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"路径不是有效文件: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        pages_text: list[str] = []
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages_text.append(f"--- 第 {idx} 页 ---\n{text}")
        full_text = "\n\n".join(pages_text)
    finally:
        doc.close()

    if not full_text.strip():
        raise ValueError("未能从 PDF 中提取到有效文本（可能是纯扫描图像 PDF，需配合 OCR 或视觉模型）。")

    return full_text


def build_prompt(case_text: str) -> tuple[str, str]:
    """构建医学实体抽取的系统提示词与用户提示词。"""
    system_prompt = (
        "你是一名严谨的临床医学信息抽取助手。"
        "请仅根据提供的病例文本抽取医学实体，严禁编造或推测原文未明确提及的信息；"
        "若某字段在文本中无明确信息，请使用 null 或空数组 []。"
        "你必须严格输出合法的 JSON 对象，不要包含 Markdown 代码块标记或多余解释。"
    )

    user_prompt = f"""请从以下中文病例文本中抽取医学实体，并严格按照指定的 JSON 结构返回：

{{
  "患者基本信息": {{
    "性别": "字符串或 null",
    "年龄": "字符串或数字或 null",
    "其他基本信息": ["字符串"]
  }},
  "主要症状": [
    "字符串（含关键表现、部位、程度或时间特征，如原文提及）"
  ],
  "既往史": {{
    "既往疾病": ["字符串"],
    "用药史": ["字符串"],
    "手术及介入史": ["字符串"],
    "过敏及传染病史": ["字符串"]
  }},
  "诊断结果": {{
    "初步或既往诊断": ["字符串"],
    "最终或核心诊断": ["字符串"],
    "关键辅助检查与阳性发现": ["字符串"]
  }},
  "治疗方案": {{
    "药物治疗": ["字符串"],
    "介入或手术治疗": ["字符串"],
    "支持与对症治疗": ["字符串"],
    "随访与后续管理": ["字符串"]
  }}
}}

病例文本如下：
{case_text}
"""
    return system_prompt, user_prompt


def parse_json_response(raw_content: str) -> dict[str, Any]:
    """解析模型返回的 JSON 文本，兼容可能的 ```json 包裹。"""
    cleaned = raw_content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是合法的 JSON: {exc}\n原始输出:\n{raw_content}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("模型返回的 JSON 顶层结构必须是对象（dict）。")

    return parsed


def extract_entities_with_qwen(
    case_text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """从环境变量读取 API Key，调用阿里云百炼 OpenAI 兼容接口抽取实体。"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到环境变量 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。请先在 .env 或系统环境中设置后再运行。"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt, user_prompt = build_prompt(case_text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回内容为空。")

    return parse_json_response(content)


def save_standardized_json(data: dict[str, Any], output_path: Path, source_pdf: Path, model: str) -> None:
    """将抽取结果封装并保存为标准化的 UTF-8 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    standardized_payload = {
        "source_file": str(source_pdf),
        "model": model,
        "entities": {
            "患者基本信息": data.get("患者基本信息") or {},
            "主要症状": data.get("主要症状") or [],
            "既往史": data.get("既往史") or {},
            "诊断结果": data.get("诊断结果") or {},
            "治疗方案": data.get("治疗方案") or {},
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(standardized_payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PyMuPDF 与阿里云百炼 Qwen 接口从中文病例 PDF 中抽取医学实体。"
    )
    parser.add_argument("--pdf", required=True, type=Path, help="输入病例 PDF 文件路径")
    parser.add_argument("--output", required=True, type=Path, help="输出标准化 JSON 文件路径")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="可选的 .env 文件路径（默认自动查找当前工作目录及父级目录中的 .env）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"百炼模型名称（默认: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"百炼 OpenAI 兼容接口 Base URL（默认: {DEFAULT_BASE_URL}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        load_env_file(args.env_file)
        case_text = extract_pdf_text(args.pdf)
        entities = extract_entities_with_qwen(
            case_text=case_text,
            model=args.model,
            base_url=args.base_url,
        )
        save_standardized_json(
            data=entities,
            output_path=args.output,
            source_pdf=args.pdf,
            model=args.model,
        )
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1

    print(f"抽取完成，已保存至: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
