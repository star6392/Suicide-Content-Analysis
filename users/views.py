from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel, TweetResultModel
from .utility.GetTweetTypes import ProcesAndDetect
from .utility.TextCatagoristion import ProcessTextCatagorisation
from .utility.BuildCNNMyModels import BuildCNNMyModel
import praw
import datetime
from django.conf import settings
# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})
def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def UserTweetOneTestForm(request):
    return render(request, 'users/UserTweetOneTestForm.html',{})

def UserTweetOneProcess(request):
    if request.method=='POST':
        tweet = request.POST.get('Tweet')
        obj = ProcesAndDetect()
        prediction,probobility = obj.preProcess(tweet)
        print(prediction,probobility)
        obj.detectTypes(tweet)
        usrname  = request.session['loginid']
        email = request.session['email']
        TweetResultModel.objects.create(usrname=usrname, email=email, tweetmsg=tweet, prediction=prediction, accuracy=probobility)
    return render(request,"users/TweeOneResults.html",{'tweet':tweet,"pred":prediction,"score":probobility})

def UserDatasetViewPraw(request):
    obj = ProcessTextCatagorisation()
    reddit = praw.Reddit(client_id='aiqVA4xtFTCiPw', client_secret='WEcypw3TBoJNhM38WxCxVEcc1lk',
                         user_agent='KushReddit', username='kushg18', password='Mygood-18')

    obj.extractMentalHealth(reddit)
    print("Mental Health Done")
    obj.extractSuicidalWatch(reddit)
    print("Suicide Watch Done")
    path = settings.MEDIA_ROOT + "\\" + "mentalHealth.txt"
    obj.wordCloud(path)
    path = settings.MEDIA_ROOT + "\\" + "suicidewatch.txt"
    obj.wordCloud2(path)
    path = settings.MEDIA_ROOT + "\\" + "data.csv"
    import pandas as pd
    df = pd.read_csv(path)
    df = df.to_html
    return render(request,"users/viewDataset.html",{'data':df})

def BuildCNNModel(request):
    obj = BuildCNNMyModel()
    accuracy = obj.startProcess()
    return render(request,"users/CNNmodelResult.html",{'accuracy':round(accuracy,3)})

def UserSearchHistoryResults(request):
    usrname = request.session['loginid']
    data = TweetResultModel.objects.filter(usrname=usrname)
    return render(request,"users/UserSearchHistory.html",{"data":data})

