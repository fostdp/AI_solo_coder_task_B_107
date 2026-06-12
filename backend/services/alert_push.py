import httpx
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class WeComPushService:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or settings.WECOM_WEBHOOK_URL
        self.client = httpx.AsyncClient(timeout=30.0)

    async def send_markdown(self, title: str, content: str, mentioned_list: Optional[List[str]] = None) -> bool:
        if not self.webhook_url or "YOUR_KEY_HERE" in self.webhook_url:
            logger.warning(f"企业微信Webhook未配置，模拟推送: {title}")
            return True

        md_text = f"## {title}\n\n{content}\n\n---\n推送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": md_text,
                "mentioned_list": mentioned_list or ["@all"],
            },
        }

        try:
            resp = await self.client.post(self.webhook_url, json=payload)
            result = resp.json()
            if result.get("errcode") == 0:
                logger.info(f"企业微信推送成功: {title}")
                return True
            else:
                logger.error(f"企业微信推送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"企业微信推送异常: {e}")
            return False

    async def send_alert(self, alert_data: Dict) -> bool:
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟠",
            "info": "🟡",
        }
        emoji = severity_emoji.get(alert_data.get("severity", "warning"), "🟠")

        title = f"{emoji} 壁画监测告警 [{alert_data.get('severity', 'warning').upper()}]"

        metrics_str = ""
        metrics = alert_data.get("metrics", {})
        if metrics:
            metrics_str = "\n**关键指标:**\n"
            for k, v in metrics.items():
                metrics_str += f"> - {k}: `{v}`\n"

        content = (
            f"**洞窟ID:** {alert_data.get('cave_id', 'N/A')}\n"
            f"**墙面:** {alert_data.get('surface_id', 'N/A')}\n"
            f"**告警类型:** {alert_data.get('alert_type', 'N/A')}\n"
            f"**详细信息:**\n{alert_data.get('message', '')}\n"
            f"{metrics_str}"
        )

        return await self.send_markdown(title, content)

    async def close(self):
        await self.client.aclose()


class SatelliteSMSService:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or settings.SATELLITE_SMS_API_URL
        self.api_key = api_key or settings.SATELLITE_SMS_API_KEY
        self.client = httpx.AsyncClient(timeout=60.0)

    async def send_sms(
        self,
        phone_numbers: List[str],
        message: str,
        source: str = "敦煌壁画监测系统",
    ) -> bool:
        if not self.api_url or "example.com" in self.api_url:
            logger.warning(f"卫星短信API未配置，模拟发送给 {phone_numbers}: {message[:80]}...")
            return True

        payload = {
            "api_key": self.api_key,
            "recipients": phone_numbers,
            "message": f"[{source}] {message}",
            "priority": "high",
            "acknowledge": True,
        }

        try:
            resp = await self.client.post(
                self.api_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            result = resp.json()
            if result.get("status") == "success" or result.get("code") == 0:
                logger.info(f"卫星短信发送成功: 发给 {len(phone_numbers)} 人")
                return True
            else:
                logger.error(f"卫星短信发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"卫星短信发送异常: {e}")
            return False

    async def send_alert(self, alert_data: Dict, phone_numbers: List[str]) -> bool:
        severity_zh = {
            "critical": "严重",
            "warning": "警告",
            "info": "提示",
        }
        sev = severity_zh.get(alert_data.get("severity", "warning"), "警告")

        msg = (
            f"[{sev}]壁画告警-{alert_data.get('cave_id', '?')}"
            f"/{alert_data.get('surface_id', '?')}: "
            f"{alert_data.get('alert_type', '异常')}"
        )

        return await self.send_sms(phone_numbers, msg)

    async def close(self):
        await self.client.aclose()


class AlertPushService:
    DEFAULT_PHONE_NUMBERS = ["13800000001", "13800000002", "13800000003"]

    def __init__(self):
        self.wecom = WeComPushService()
        self.satellite = SatelliteSMSService()

    async def push_alert(self, alert_data: Dict, channels: Optional[List[str]] = None) -> Dict:
        channels = channels or ["wecom", "satellite_sms"]
        results = {"wecom": False, "satellite_sms": False}

        if "wecom" in channels:
            results["wecom"] = await self.wecom.send_alert(alert_data)

        if "satellite_sms" in channels:
            phones = self.DEFAULT_PHONE_NUMBERS
            results["satellite_sms"] = await self.satellite.send_alert(alert_data, phones)

        return results

    async def close(self):
        await self.wecom.close()
        await self.satellite.close()
