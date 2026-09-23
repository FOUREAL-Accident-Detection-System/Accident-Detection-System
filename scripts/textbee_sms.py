from dotenv import load_dotenv
import os
import requests

load_dotenv()


def send_accident_alert(timestamp, confidence):
    api_key = os.getenv("TEXTBEE_API_KEY")
    recipient = os.getenv("CONTROL_ROOM_NUMBER")
    device_id = os.getenv("TEXTBEE_DEVICE_ID")

    url = "https://api.textbee.dev/api/v1/gateway/send-sms"

    message = (
        "ACCIDENT ALERT!\n"
        f"Accident detected at {timestamp:.2f} seconds.\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    payload = {
        "recipients": [recipient],
        "message": message,
        "deviceId": device_id
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30
        )

        print("HTTP status:", response.status_code)
        print("TextBee response:")
        print(response.text)

        if response.status_code == 200:
            return True

        return False

    except Exception as e:
        print("SMS sending failed.")
        print(e)
        return False


if __name__ == "__main__":
    send_accident_alert(
        timestamp=5.85,
        confidence=0.92
    )
