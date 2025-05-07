from celery import shared_task
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Medication
import requests  # For SMS API integration

@shared_task
def schedule_medication_notifications(medication_id):
    try:
        medication = Medication.objects.get(id=medication_id)
        
        if not medication.is_active or not medication.mobile_number:
            return
        
        # Calculate days between start and end date
        days = (medication.end_date - timezone.now().date()).days if medication.end_date else 365
        
        for time_str in medication.notification_times:
            hour, minute = map(int, time_str.split(':'))
            
            # Schedule for each day until end date
            for day in range(days + 1):
                target_date = timezone.now().date() + timedelta(days=day)
                send_medication_reminder.apply_async(
                    args=[medication_id],
                    eta=datetime(
                        target_date.year,
                        target_date.month,
                        target_date.day,
                        hour,
                        minute,
                        0,
                        tzinfo=timezone.get_current_timezone()
                    )
                )
                
    except Medication.DoesNotExist:
        pass

@shared_task
def send_medication_reminder(medication_id):
    try:
        medication = Medication.objects.get(id=medication_id)
        
        if not medication.is_active or not medication.mobile_number:
            return
        
        # Customize this message as needed
        message = (
            f"Medication Reminder: Take {medication.name} ({medication.dosage}). "
            f"Purpose: {medication.purpose or 'Not specified'}"
        )
        
        # Replace with your actual SMS API integration
        send_sms_notification(medication.mobile_number, message)
        
    except Medication.DoesNotExist:
        pass

def send_sms_notification(phone_number, message):
    # Example using Twilio (you'll need to install twilio package)
    try:
        from twilio.rest import Client
        
        account_sid = 'AC0c290c03b845a8f99c1b44c72b9d2a52'
        auth_token = 'ba6cc8dbc87f3cdbb438d9f0ede4ed42'
        from_number = '+15202317923'
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=phone_number
        )
        return message.sid
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return None