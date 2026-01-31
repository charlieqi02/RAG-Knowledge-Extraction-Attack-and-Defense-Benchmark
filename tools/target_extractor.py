import json
from typing import Dict, List, Union, Optional
from tools.get_llm import get_llm
from tqdm import tqdm



import json
import re


def safe_json_loads(text: str):
    """
    Robust JSON extractor for LLM outputs.
    Returns dict or None.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Case 1: pure JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Case 2: extract {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


DEFAULT_SYS_PROMPT = """
You are a data extraction expert. You will be given a text and a list of 'Known Schemas'.
Your task:
1. Extract private information from the text. 
2. Use the 'Known Schemas' where applicable. 
3. If you find private information that doesn't fit 'Known Schemas', CREATE a new concise, uppercase schema name (e.g., 'MEDICATION').
4. Return ONLY a valid JSON object where keys are the schemas and values are lists of extracted strings.
5. Do NOT include a schema key if no information for it is found in the text.


Example:
Known Schemas: NAME, EMAIL, LOCATION
Text: "John Doe from London has Diabetes."
Output: 
{
    "NAME": ["John Doe"],
    "LOCATION": ["London"],
    "DISEASE": ["Diabetes"]
}
""".strip()


class PrivacyExtractor:
    """
    Minimal reusable privacy extractor.
    Input  : documents (dict or list)
    Output : {doc_id: {SCHEMA: [values]}}
    """

    def __init__(
        self,
        llm_name: str = "gpt4o-mini",
        init_schemas: Optional[List[str]] = None,
        sys_prompt: str = DEFAULT_SYS_PROMPT,
        user_prompt_template: str = "Known Schemas: {schemas}\nText: {text}",
    ):
        self.llm = get_llm(llm_name)
        self.schemas = set(s.upper() for s in (init_schemas or []))
        self.sys_prompt = sys_prompt
        self.user_prompt_template = user_prompt_template

    def extract(
        self,
        documents: Union[Dict[str, str], List[str]],
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Main entry.
        documents:
          - Dict[id, text]
          - or List[text] (auto-id: doc_0, doc_1, ...)
        """
        results: Dict[str, Dict[str, List[str]]] = {}

        for text in tqdm(documents, desc="Extracting Privacy Info"):
            extracted = self._extract_one(text)
            if extracted:
                results[text] = extracted

        return results

    def _extract_one(self, text: str) -> Dict[str, List[str]]:
        prompt = self.user_prompt_template.format(
            schemas=", ".join(sorted(self.schemas)),
            text=text,
        )

        resp = self.llm([
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ])

        data = safe_json_loads(resp)
        normalized: Dict[str, List[str]] = {}
        if data is not None:
            for k, v in data.items():
                key = k.upper().strip()
                values = v if isinstance(v, list) else [v]
                values = [str(x) for x in values if str(x).strip()]

                if values:
                    normalized[key] = values
                    self.schemas.add(key)  # schema discovery

        return normalized
