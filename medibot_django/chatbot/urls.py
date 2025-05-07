
# from django.urls import path
# from . import views
# from django.views.generic.base import RedirectView



# urlpatterns = [
#     path('', views.chatbot_response, name='chatbot'),
#     path('api/chat/', views.chatbot_response, name='chatbot_response'),
#     path('favicon.ico', RedirectView.as_view(url='/static/images/favicon.ico', permanent=True))
# ]

# from django.urls import path
# from . import views
# from django.views.generic.base import RedirectView
# urlpatterns = [
#     path('', views.chatbot_response, name='chatbot'),  # Map root URL to chatbot_response view
# ]

# ====working level 1
# from django.urls import path
# from .views import chatbot_response
# from . import views

# urlpatterns = [
#     path('chatbot/', chatbot_response, name='chatbot_response'),
# ]

# for auth ===

# working 12-04025


# from django.urls import path
# from .views import chatbot_response, signup, user_login, user_logout

# urlpatterns = [
#     path('chatbot/', chatbot_response, name='chatbot_response'),
#     path('signup/', signup, name='signup'),
#     path('login/', user_login, name='user_login'),
#     path('logout/', user_logout, name='user_logout'),
# ]



from django.urls import path
from django.shortcuts import redirect
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    chatbot_response, 
    signup, 
    user_login, 
    user_logout, 
    dashboard_view, 
    medical_assistant,
    doctor_dashboard_view,
    appointment_view,
    find_doctor_view,
    book_appointment_view,
    manage_appointments_view,
    medication_view, 
    medication_toggle, 
    medication_delete,
    medical_records_view,
    upload_medical_record,
    create_prescription,
    health_tips_view,
    save_tip,
    saved_tips_view,

)

def redirect_to_signup(request):
    if request.user.is_authenticated:
        if request.user.user_type == 'doctor':
            return redirect('doctor_dashboard')
        return redirect('dashboard')
    else:
        return redirect('signup')

urlpatterns = [
    path('', redirect_to_signup, name='root'),
    path('chatbot/', chatbot_response, name='chatbot_response'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('doctor_dashboard/', doctor_dashboard_view, name='doctor_dashboard'),
    path('chat_assistant/', medical_assistant, name='medical_assistant'),
    path('signup/', signup, name='signup'),
    path('login/', user_login, name='user_login'),
    path('logout/', user_logout, name='user_logout'),
    
    # New URLs for appointment system
    path('appointments/', appointment_view, name='appointments'),
    path('find-doctors/', find_doctor_view, name='find_doctors'),
    path('book-appointment/<int:doctor_id>/', book_appointment_view, name='book_appointment'),
    path('manage-appointments/', manage_appointments_view, name='manage_appointments'),
    path('medications/', medication_view, name='medications'),
    path('medications/<int:med_id>/toggle/', medication_toggle, name='medication_toggle'),
    path('medications/<int:med_id>/delete/', medication_delete, name='medication_delete'),
    path('medical_records/',medical_records_view, name='medical_records'),
    path('upload_medical_record/',upload_medical_record, name='upload_medical_record'),
    path('create-prescription/', create_prescription, name='create_prescription'),
    path('health-tips/', health_tips_view, name='health_tips'),
    path('save-tip/<int:tip_id>/', save_tip, name='save_tip'),
    path('saved-tips/', saved_tips_view, name='saved_tips'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)