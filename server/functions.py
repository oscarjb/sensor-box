import random
from .models import *


def check_pin(doc):
    num = random.randrange(1, 10**6)
    num_with_zeros = '{:06}'.format(num)
    num_with_zeros = str(num).zfill(6)
    if len(Doctor.objects.all().filter(pin=num_with_zeros))>0:
        check_pin(doc)
    else:
        doc.pin = num_with_zeros
        doc.save()
    return