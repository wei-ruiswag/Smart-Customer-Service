import json
import requests

url = "http://127.0.0.1:8000/api/tools/call"

payload = {
    "name": "ticket_create",
    "arguments": {
        "user_id": "6",
        "order_no": "ORD202606070001",
        "ticket_type": "退款",
        "priority": "中",
        "description": "用户申请退款"
    }
}

resp = requests.post(url, json=payload, timeout=60)

print("status_code:", resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))