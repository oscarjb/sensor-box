from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import *
from django.contrib import auth
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.template.context_processors import csrf
from django.views.decorators.csrf import csrf_exempt
from .forms import *
import json
import ast
from .model_functions import *
import os

import PIL 
from PIL import Image





@login_required(login_url='login_register')
def home(request):
    context = {}
    context['profile'] = Doctor.objects.all().get(user=request.user)
    return render(request, "index.html", context)

@login_required(login_url='login_register')
def profile(request):
    context = {}
    doctor = Doctor.objects.all().get(user=request.user)
    context['profile'] = doctor
    context['patients'] = Patient.objects.all().filter(doctor=doctor)
    return render(request, "profile.html", context)

@login_required(login_url='login_register')
def search_users(request, patient_id):
    context = {}
    context['profile'] = Doctor.objects.all().get(user=request.user)
    context['patient'] = Patient.objects.all().get(id=patient_id)
    context['tests'] = Test.objects.all().filter(patient_id=patient_id)
    #print(context)
    return render(request, "patient_profile.html", context)

@login_required(login_url='login_register')
def search_tests(request, patient_id, test_id):
    
    context = {}
    context['profile'] = Doctor.objects.all().get(user=request.user)
    context['patient'] = Patient.objects.all().get(id=patient_id)
    test = Test.objects.all().get(id=test_id)
    context['test'] = test
    
    data_test = {}
    print(test.camera_kurokesu)

    try:
        data_gas = ast.literal_eval(test.gas_sensors)
        colours = ["rgba(255, 0, 0, 0.5)", "rgba(0, 255, 0, 0.5)", "rgba(0, 0, 255, 0.5)", "rgba(255, 0, 255, 0.5)"]
        cont = 0
        # print(data_gas)
        for key in data_gas.keys():
            if key!="HCSR04":
                data = data_gas[key]
                ppms = []
                for i in data:
                    try:
                        ppms.append(float(i[0]))
                    except:
                        pass
                data_test[key] = [ppms, colours[cont]]
                cont+=1
        labels = [i for i in range(1, len(ppms))]
        context['data_test'] = data_test
        context['label'] = labels
    except Exception as e:
        print(e)
        pass
    return render(request, "test_overview2.html", context)


@login_required(login_url='login_register')
def edit_user(request):
    context = {}
    doctor = Doctor.objects.all().get(user=request.user)
    context['profile'] = Doctor.objects.all().get(user=request.user)
    if request.method == 'POST':
        form = User_Doctor_Form(request.POST, request.FILES, instance={
            'user': request.user,
            'doctor': doctor,
        })
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/')
    else:
        form = User_Doctor_Form
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return render(request, "edit_user.html", context)

@login_required(login_url='login_register')
def edit_patient(request, patient_id):
    context = {}
    doctor = Doctor.objects.all().get(user=request.user)
    patient = Patient.objects.all().get(id=patient_id)
    context['profile'] = doctor
    context['patient'] = patient
    if request.method == 'POST':
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            birthdate = "" + request.POST.get("year") + "-" + request.POST.get("month") + "-" + request.POST.get("day")
            if request.FILES.get('photo') != None:
                photo = request.FILES.get('photo')
                patient.photo = photo
            patient.birth_date = birthdate
            patient.name = form.cleaned_data['name']
            patient.gender = form.cleaned_data['gender']
            patient.save()
            return search_users(request, patient_id)
    else:
        form = PatientForm
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return  render(request, 'edit_patient.html', context)

@login_required(login_url='login_register')
def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/login/#signin")


def login_register(request):
    context = {}
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/login/#signin')
        else:
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_active:
                    if Doctor.objects.filter(user=user).count() == 0:
                        doctor = Doctor(user=user)
                        doctor.save()
                    login(request, user)
                    return HttpResponseRedirect('/profile/')
    else:
        form = UserForm
    token = {}
    token.update(csrf(request))
    token['form'] = form
    return render(request, "login.html", context)

@csrf_exempt
def login_api(request):
    if request.method == 'POST':
        pin = request.POST['pin']
        response = {}
        response['status'] = "missing"
        response['patients'] = []
        if len(Doctor.objects.filter(pin=pin)) > 0:
            doctor = Doctor.objects.all().get(pin=pin)
            patients = Patient.objects.filter(doctor=doctor)
            for patient in patients:
                temp = [patient.id, patient.name]
                response['patients'].append(temp)
            response['status'] = "correct"
        else:
            response['status'] = "missing"
        return JsonResponse(response)

@login_required(login_url='login_register')
def create_patient(request):
    context = {}
    doctor = Doctor.objects.all().get(user=request.user)
    context['profile'] = doctor
    if request.method == 'POST':
        existe = Patient.objects.all().filter(name=request.POST.get('name'),
                                               doctor=doctor).exists()
        if not existe:
            birthdate = ""+ request.POST.get("year") + "-" + request.POST.get("month") + "-" + request.POST.get("day")
            photo = request.FILES.get('photo')
            if photo is not None:
                patient = Patient(name=request.POST.get('name'), gender=request.POST.get('gender'),
                                  birth_date=birthdate, doctor=doctor, photo=photo)
            else:
                patient = Patient(name=request.POST.get('name'), gender=request.POST.get('gender'),
                                  birth_date=birthdate, doctor=doctor)
            patient.save()
            return search_users(request, patient.id)
        else:
            patient = Patient.objects.all().get(name=request.POST.get('name'),
                                               doctor=doctor)
            return search_users(request, patient.id)

    token = {}
    token.update(csrf(request))
    return render(request, 'create_patient.html', context)


@login_required(login_url='login_register')
def remove_patient(request, patient_id):
    patient = Patient.objects.all().get(id=patient_id)
    patient.delete()
    return HttpResponseRedirect('/profile/')

@login_required(login_url='login_register')
def remove_test(request, patient_id, test_id):
    test = Test.objects.all().get(id=test_id)
    test.delete()
    return search_users(request, patient_id)

@csrf_exempt
def recieve_data(request):
    if request.method == 'POST':
        # merged = {**request.POST, **request.FILES}
        response = {}
        response['status'] = "failed"
        doctor = Doctor.objects.all().get(pin=request.POST['pin'])
        try:
            data_test={
                "user": doctor
            }
            for key in request.POST:
                if key != 'pin': data_test[key] = request.POST.get(key)
            for key in request.FILES:
                data_test[key] = request.FILES.get(key)
            test = Test.objects.create(**data_test)
            test.outcome = "Negative"
            data_gas = ast.literal_eval(test.gas_sensors)
            for key in data_gas.keys():
                if key != "HCSR04":
                    data = data_gas[key]
                    ppms = []
                    for i in data:
                        try:
                            ppms.append(float(i[0]))
                        except:
                            pass
                    if predict(ppms) == 0: test.outcome = "Positive"
            test.TissueTypes ="None"

            test.Ruler ="No"
            test.ImageRight = "Image"
            test.ImageLeft = "Image"
            test.Imagetherm = "Image"
            test.Perimeter = "0 mm"
            test.Area = "0 mm2"
            test.Granulation = "0 %"
            test.Slough = "0 %"
            test.Necrosis = "0 %"
            test.save()
            response['status'] = "success"
        except Exception as e:
            response['status'] = "Exception ocurred - " + str(e)
        return  JsonResponse(response)

def analyze_data(request):
    #if request.method == 'POST':
    test_id = request.GET.get('test_id', None)
    test = Test.objects.all().get(id=test_id)
    print(test)
    image2 = Image.open(os.path.join('server/static/images/', str(test.camera_kurokesu).replace('\\', '/') ))
    width, height = image2.size
    image2 = image2.resize((int(width/4), int(height/4)), PIL.Image.ANTIALIAS)
    response_thread = segment(np.array(image2),test)
    print(response_thread)
    if (response_thread["status"] == "Finished"):
        return JsonResponse(response_thread)
    else:
        context = {"status": "working"}
        return JsonResponse(context)
