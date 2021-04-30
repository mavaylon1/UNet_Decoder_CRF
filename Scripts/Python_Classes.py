# methods are functions associated with classes

#If you want to create an empty class:

class Employee:
    pass

# --------------------------------------------------
# Class vs instance of a class
# --------------------------------------------------

"""
Emp1 and emp2 are unique "instances" of the employee class.

When we print them, we will see that they are "Employee Objects"
"""
emp1 = Employee()
emp2 = Employee()

print(emp1)
print(emp2)

# --------------------------------------------------
# Instance Variables
# --------------------------------------------------

"""
These are variables unique to each instance of a class.

We can set these manually (not recommended) as follows.
"""
emp1.first = 'corey'
emp1.last = 'smith'
emp1.email = 'something@email.com'
emp1.pay = 50000

emp2.first = 'test'
emp2.last = 'user'
emp2.email = 'test@email.com'
emp2.pay = 60000

# We can do this in a better way by having it defined when creating an instance.
# Let's create a new class
class Employee2:

    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

# Let's create an instance of Employee2

emp1 = Employee2('jim','bob',30000)

# IMPORTANT: in this example above. the "first" "last" "pay" "email" are all "instance attributes"
# or "instance variables" of our class

# --------------------------------------------------
# methods
# --------------------------------------------------

class Employee3:

    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

emp1 = Employee3('jim','bob',30000)

print(emp1.fullname())
# this will return the full name

# --------------------------------------------------
# Class Variables
# --------------------------------------------------

"""
Class variables are variables shared/the same among every instance of the class.

Whereas instance variables are unique to each instance.
"""

class Employee4:

    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * 1.04)

emp1 = Employee4('jim','bob',30000)

print(emp1.pay)
# return 30000
emp1.apply_raise()
print(emp1.pay)
# return 31200

# What's wrong with this?
# Well we can't actually see the raise amount, nor can we change it.
# to solve this, we can set it as a class variable
class Employee5:
    raise_amount = 1.04
    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * Employee5.raise_amount)
        # or
        # self.pay = int(self.pay * self.raise_amount)
        # Self referes to the instance of the class so how is it possible
        # that a class variable can be accessed by "self.raise_amount"

emp1 = Employee5('jim','bob',30000)
emp2 = Employee5('hi','bye',4000)


print(emp1.__dict__)
#This returns all the instance variables and as you can see raise_amount is NOT There

print(Employee5.__dict__)
# Raise amount is There

# What Python does is that it searches to see if the instance variable exists, and if doesn't
# then it'll check the class variables.

Employee5.raise_amount = 1.05
#This will change the raise amout for the class and for every instances

emp1.raise_amount = 1.05
#This changes the raise amount for just emp1. Why?
print(emp1.__dict__)
# What we now see is that it created an instance variable for raise_amount and set it to 1.05
# this doesn't change emp2


# --------------------------------------------------
# Classmethods and Staticmethods
# --------------------------------------------------
"""
Regular methods in a class automatically take the instance as the first argument and we called it
"self".

But what if we want to make the class as the first argument?
Well we add the decorator "@classmethod before it and use "cls" instead of "self"
"""

class Employee6:

    raise_amount = 1.04

    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * Employee5.raise_amount)
        # or
        # self.pay = int(self.pay * self.raise_amount)
        # Self referes to the instance of the class so how is it possible
        # that a class variable can be accessed by "self.raise_amount"

    @classmethod
    def set_raise_amount(cls, amount):
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True

emp1 = Employee6('jim','bob',30000)
emp2 = Employee5('hi','bye',4000)

print(emp1.raise_amount) #1.04
print(emp2.raise_amount)#1.04

Employee6.set_raise_amount(1.05)
#This is equal to what we did before as
# Employee5.raise_amount = 1.05


print(emp1.raise_amount) #1.05
print(emp2.raise_amount)#1.05

# Our class method set_raise_amount changed the class variable raise_amount for the class
# which affects every instance of the class

"""
Some people use classmethods as alternative constructors
For example, say that people have a set of strings that have the new employee
information "John-Doe-3000". To use our class we would have to seperate out the strings
, but what if we had a class method that did it for us. By calling the class within this classmethod
the strings will be fed into the class constructor and create an new Employee object.
"""
emp3 = Employee6.from_string("John-Doe-3000")

# ---
# What about Staticmethods??
# ---
"""
Well let's review a quick moment.

- Regular class methods automatically feed in the instance and that's why we have self.
- Class methods feed in the class automatically and that's why we have cls.

Staticmethods don't feed in anything. And behave like regular functions.

When would we use them? Well we would use staticmethod when the method is related to class, But
does not reference the class or an instance within the body of the method. For example, in our Employee6 class,
we see that "is_workday" is our staticmethod and all it does is return whether the day of the week is a workday.
Note: python has a built in function that says str.workday() and returns a number between [0=Monday, ...6=Sunday].

There is no where we see "cls" or "self" in "is_workday"; therefore, we should make it a 'staticmethod'.
"""

# --------------------------------------------------
# Inheritance
# --------------------------------------------------

class Employee7:

    raise_amount = 1.04

    def __init__(self, first, last, pay):
        #Think of this as the constructor
        self.first = first
        # the self refers to the instance of the class.
        # saying self.first = first is the equivalent of what we did before of emp1.first = "corey
        # except now it's created when we create an instance
        self.last = last
        self.pay = pay
        self.email = first + "." + last + '@company.com'

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * Employee5.raise_amount)

"""
Even though it inherits class variables, instance variables, and methods from Employee7, we can change them.

For example we can change the raise_amount and it will no longer use the parent class Employee raise amount, but won't
affect the parent class.

What if we want developer to have more instance variables than what was inherited. Remember that the child class inherits the
constructor of the parent class, so we would then create a new constructor for the developer:conda condy
yy.
"""
class developer(Employee7):
    raise_amount = 1.10

# developer inherits the instance variables in the parent constructor.

# A useful function is help
print(help(developer))
# ---> This will show the "Method Resolution order", which tells you where python is searching for variables and methods in order.
