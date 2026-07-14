# Copyright (c) 2026, Norwa Group and Contributors
# See license.txt

# import frappe
from unittest.mock import patch

from frappe.tests.utils import FrappeTestCase

from changai.changai.api.v2 import clients


class TestChangAISettings(FrappeTestCase):
    @patch.object(clients, "call_gemini", return_value="gemini response")
    @patch.object(clients.ChangAIConfig, "get", return_value={"llm": "Gemini"})
    def test_routes_gemini_provider(self, _get_config, call_gemini):
        response = clients.call_model("hello", sys_prompt="system")

        self.assertEqual(response, "gemini response")
        call_gemini.assert_called_once_with("hello", "system")

    @patch.object(clients, "local_llm_request", return_value="ollama response")
    @patch.object(clients.ChangAIConfig, "get", return_value={"llm": "Ollama"})
    def test_routes_ollama_provider(self, _get_config, local_llm_request):
        response = clients.call_model("hello", sys_prompt="system")

        self.assertEqual(response, "ollama response")
        local_llm_request.assert_called_once_with("hello", "system")

    @patch.object(clients, "remote_llm_request_deploy_test", return_value="qwen response")
    @patch.object(
        clients.ChangAIConfig,
        "get",
        return_value={
            "llm": "QWEN3",
            "deploy_url": "https://api.example.test/deployments/qwen/predictions",
            "API_TOKEN": "token",
        },
    )
    def test_routes_qwen_provider(self, _get_config, remote_request):
        response = clients.call_model("hello", task="llm", sys_prompt="system")

        self.assertEqual(response, "qwen response")
        remote_request.assert_called_once_with(prompt="hello", task="llm")

    @patch.object(
        clients.ChangAIConfig,
        "get",
        return_value={
            "ollama_url": "http://127.0.0.1:11434/",
            "ollama_model": "qwen3:4b",
        },
    )
    @patch.object(
        clients,
        "_post_json",
        return_value={"ok": True, "status_code": 200, "body": {"response": " local response "}},
    )
    def test_ollama_generate_payload(self, post_json, _get_config):
        response = clients.local_llm_request("hello", "system")

        self.assertEqual(response, "local response")
        post_json.assert_called_once_with(
            "http://127.0.0.1:11434/api/generate",
            headers={},
            payload={
                "model": "qwen3:4b",
                "prompt": "hello",
                "stream": False,
                "system": "system",
            },
            timeout=120,
        )
