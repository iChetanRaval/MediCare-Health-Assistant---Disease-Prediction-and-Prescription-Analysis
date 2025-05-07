# from django.db import models

# # Create your models here.

# for auth

# from django.db import models
# from django.contrib.auth.models import AbstractUser
# from django.utils import timezone

# class CustomUser(AbstractUser):
#     USER_TYPE_CHOICES = (
#         ('patient', 'Patient'),
#         ('doctor', 'Doctor'),
#     )
#     user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='patient')
    
#     # Doctor specific fields
#     doctor_id = models.CharField(max_length=50, blank=True, null=True)
#     specialization = models.CharField(max_length=100, blank=True, null=True)
#     experience = models.PositiveIntegerField(blank=True, null=True)
#     clinic_address = models.TextField(blank=True, null=True)
#     contact_number = models.CharField(max_length=20, blank=True, null=True)
#     full_name = models.CharField(max_length=100, blank=True, null=True)
#     profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)

#     def __str__(self):
#         return self.username

#     def is_doctor(self):
#         return self.user_type == 'doctor'

#     def is_patient(self):
#         return self.user_type == 'patient'

# class Appointment(models.Model):
#     STATUS_CHOICES = (
#         ('pending', 'Pending'),
#         ('confirmed', 'Confirmed'),
#         ('cancelled', 'Cancelled'),
#         ('completed', 'Completed'),
#     )
    
#     patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='patient_appointments')
#     doctor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='doctor_appointments')
#     date = models.DateField()
#     time = models.TimeField()
#     reason = models.TextField()
#     status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     class Meta:
#         ordering = ['date', 'time']

#     def __str__(self):
#         return f"Appointment with Dr. {self.doctor} on {self.date} at {self.time}"
    


# from django.db import models
# from django.contrib.auth.models import User
# from django.utils import timezone

# class Medication(models.Model):
#     FREQUENCY_CHOICES = [
#         (1, 'Once daily'),
#         (2, 'Twice daily'),
#         (3, 'Three times daily'),
#         (4, 'Four times daily'),
#         (0, 'As needed'),
#     ]

#     patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='medications')
#     name = models.CharField(max_length=100)
#     purpose = models.CharField(max_length=100, blank=True, null=True)
#     dosage = models.CharField(max_length=50)
#     frequency = models.IntegerField(choices=FREQUENCY_CHOICES)
#     start_date = models.DateField(default=timezone.now)
#     end_date = models.DateField(blank=True, null=True)
#     notes = models.TextField(blank=True, null=True)
#     created_at = models.DateTimeField(auto_now_add=True)
#     is_active = models.BooleanField(default=True)

#     def __str__(self):
#         return f"{self.name} ({self.dosage})"

#     @property
#     def status(self):
#         if not self.is_active:
#             return "inactive"
#         if self.end_date and self.end_date < timezone.now().date():
#             return "expired"
#         if self.end_date and (self.end_date - timezone.now().date()).days <= 3:
#             return "ending_soon"
#         return "active"

# class CommonMedication(models.Model):
#     name = models.CharField(max_length=100)
#     condition = models.CharField(max_length=100)
#     dosage = models.CharField(max_length=100)
#     frequency = models.CharField(max_length=100)
#     instructions = models.TextField()

#     def __str__(self):
#         return f"{self.name} for {self.condition}"
    

# from django.db import models
# from django.core.validators import FileExtensionValidator

# class Prescription(models.Model):
#     appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='prescriptions')
#     doctor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='doctor_prescriptions')
#     patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='patient_prescriptions')
#     medication = models.CharField(max_length=100)
#     dosage = models.CharField(max_length=50)
#     instructions = models.TextField()
#     prescribed_date = models.DateField(default=timezone.now)
#     duration = models.CharField(max_length=50, help_text="e.g., 10 days, 2 weeks, etc.")
#     is_active = models.BooleanField(default=True)

#     def __str__(self):
#         return f"{self.medication} for {self.patient.username}"

# class MedicalRecord(models.Model):
#     RECORD_TYPE_CHOICES = [
#         ('lab', 'Lab Report'),
#         ('imaging', 'Imaging Report'),
#         ('procedure', 'Procedure Note'),
#         ('progress', 'Progress Note'),
#         ('other', 'Other'),
#     ]
    
#     patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='medical_records')
#     record_type = models.CharField(max_length=20, choices=RECORD_TYPE_CHOICES)
#     title = models.CharField(max_length=200)
#     date = models.DateField(default=timezone.now)
#     doctor = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_records')
#     file = models.FileField(
#         upload_to='medical_records/',
#         validators=[FileExtensionValidator(['pdf', 'jpg', 'jpeg', 'png'])]
#     )
#     notes = models.TextField(blank=True, null=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"{self.get_record_type_display()} - {self.title}"



from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator

def validate_duration(value):
    if not any(word in value.lower() for word in ['day', 'week', 'month', 'year']):
        raise ValidationError("Duration should include time unit (e.g., 'days', 'weeks')")

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = (
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='patient')
    
    # Doctor specific fields
    doctor_id = models.CharField(max_length=50, blank=True, null=True)
    specialization = models.CharField(max_length=100, blank=True, null=True)
    experience = models.PositiveIntegerField(blank=True, null=True)
    clinic_address = models.TextField(blank=True, null=True)
    contact_number = models.CharField(max_length=20, blank=True, null=True)
    full_name = models.CharField(max_length=100, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)

    def __str__(self):
        return self.username

    def is_doctor(self):
        return self.user_type == 'doctor'

    def is_patient(self):
        return self.user_type == 'patient'

class Appointment(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled'),
        ('completed', 'Completed'),
    )
    
    patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='patient_appointments')
    doctor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='doctor_appointments')
    date = models.DateField()
    time = models.TimeField()
    reason = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['date', 'time']

    def __str__(self):
        return f"Appointment with Dr. {self.doctor} on {self.date} at {self.time}"

class Medication(models.Model):
    FREQUENCY_CHOICES = [
        (1, 'Once daily'),
        (2, 'Twice daily'),
        (3, 'Three times daily'),
        (4, 'Four times daily'),
        (0, 'As needed'),
    ]

    patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='medications')
    name = models.CharField(max_length=100)
    purpose = models.CharField(max_length=100, blank=True, null=True)
    dosage = models.CharField(max_length=50)
    frequency = models.IntegerField(choices=FREQUENCY_CHOICES)
    start_date = models.DateField(default=timezone.now)
    end_date = models.DateField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.dosage})"

    @property
    def status(self):
        if not self.is_active:
            return "inactive"
        if self.end_date and self.end_date < timezone.now().date():
            return "expired"
        if self.end_date and (self.end_date - timezone.now().date()).days <= 3:
            return "ending_soon"
        return "active"

class CommonMedication(models.Model):
    name = models.CharField(max_length=100)
    condition = models.CharField(max_length=100)
    dosage = models.CharField(max_length=100)
    frequency = models.CharField(max_length=100)
    instructions = models.TextField()

    def __str__(self):
        return f"{self.name} for {self.condition}"

class Prescription(models.Model):
    PRESCRIPTION_STATUS = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('expired', 'Expired'),
    ]
    
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='prescriptions')
    doctor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='doctor_prescriptions')
    patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='patient_prescriptions')
    medication = models.CharField(max_length=100)
    dosage = models.CharField(max_length=50)
    instructions = models.TextField()
    prescribed_date = models.DateField(default=timezone.now)
    duration = models.CharField(max_length=50, help_text="e.g., 10 days, 2 weeks, etc.", 
                              validators=[validate_duration])
    refill = models.PositiveIntegerField(default=0, help_text="Number of refills remaining")
    status = models.CharField(max_length=10, choices=PRESCRIPTION_STATUS, default='active')
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.medication} for {self.patient.username}"

class MedicalRecord(models.Model):
    RECORD_TYPE_CHOICES = [
        ('lab', 'Lab Report'),
        ('imaging', 'Imaging Report'),
        ('procedure', 'Procedure Note'),
        ('progress', 'Progress Note'),
        ('other', 'Other'),
    ]
    
    patient = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='medical_records')
    record_type = models.CharField(max_length=20, choices=RECORD_TYPE_CHOICES)
    title = models.CharField(max_length=200)
    date = models.DateField(default=timezone.now)
    doctor = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_records')
    file = models.FileField(
        upload_to='medical_records/',
        validators=[FileExtensionValidator(['pdf', 'jpg', 'jpeg', 'png'])]
    )
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_record_type_display()} - {self.title}"


# models.py
from django.db import models
from django.contrib.auth.models import User

class HealthTipCategory(models.Model):
    name = models.CharField(max_length=100)
    icon = models.CharField(max_length=50, default='lightbulb')
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

from django.conf import settings
from django.db import models
from django.utils.text import slugify

class HealthTip(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True, blank=True)  # Add this field
    content = models.TextField()
    category = models.ForeignKey(HealthTipCategory, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_featured = models.BooleanField(default=False)
    source = models.CharField(max_length=200, blank=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

class SavedTip(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)  # Updated this line
    tip = models.ForeignKey(HealthTip, on_delete=models.CASCADE)
    saved_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'tip')

class WellnessArticle(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    summary = models.TextField()
    read_time = models.PositiveIntegerField(help_text="Estimated reading time in minutes")
    published_date = models.DateField()
    author = models.CharField(max_length=100, blank=True)
    image = models.ImageField(upload_to='articles/', blank=True, null=True)

    def __str__(self):
        return self.title