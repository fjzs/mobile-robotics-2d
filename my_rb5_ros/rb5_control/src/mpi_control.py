#!/usr/bin/env python
"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from megapi import MegaPi


MFR = 2     # port for motor front right
MBL = 3     # port for motor back left
MBR = 10    # port for motor back right
MFL = 11    # port for motor front left


class MegaPiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR  # port for motor front right
        self.mbl = MBL  # port for motor back left
        self.mbr = MBR  # port for motor back right
        self.mfl = MFL  # port for motor front left   

    
    def printConfiguration(self):
        print('MegaPiController:')
        print("Communication Port:" + repr(self.port))
        print("Motor ports: MFR: " + repr(MFR) +
              " MBL: " + repr(MBL) + 
              " MBR: " + repr(MBR) + 
              " MFL: " + repr(MFL))


    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0):
        if self.verbose:
            print("Set Motors: vfl: " + repr(int(round(vfl,0))) + 
                  " vfr: " + repr(int(round(vfr,0))) +
                  " vbl: " + repr(int(round(vbl,0))) +
                  " vbr: " + repr(int(round(vbr,0))))

        spd1 = 0.99
        spd2 = -0.000575*vfr + 1.11725
        spd3 = 0.98
        spd4 = -0.000575*vbr + 1.11725
        
        # this is front right motor
        self.bot.motorRun(self.mfl,vfl*spd1)
        # this is back left motor
        self.bot.motorRun(self.mfr,vfr*spd2)
        # this is back right motor
        self.bot.motorRun(self.mbl,vbl*spd3)
        # this is front left motor
        self.bot.motorRun(self.mbr,vbr*spd4)


    def carStop(self):
        if self.verbose:
            print("CAR STOP:")
        self.setFourMotors()


    def carStraight(self, speed):
        if self.verbose:
            print("CAR STRAIGHT:")
        #self.setFourMotors(-speed, speed, -speed, speed) #old default setting
        # self.setFourMotors(speed, -speed, speed, -speed)
        self.setFourMotors(speed, -speed, speed, -speed)


    def carRotate(self, speed):
        if self.verbose:
            print("CAR ROTATE:")
        #self.setFourMotors(-speed, -speed, -speed, -speed) #old default setting
        self.setFourMotors(-speed, -speed, -speed, -speed)

    def carRotateWithDegree(self, speed, degree, time):
        if self.verbose:
            print("CAR ROTATE:")
        #self.setFourMotors(-speed, -speed, -speed, -speed) #old default setting
        self.setFourMotors(-speed, -speed, -speed, -speed)
        
        # calculate cc degree:
        if degree <= -360 or degree <=360:
            delayT = degree * 2.1/360.0
            print ("The delay time is: ", delayT)

            if speed < 100:
                offset = abs(speed * 2.104/100.0)
            else:
                offset = 0
            print("The rotation offset time is: ", offset, "and Delay time is: ", delayT)
            time.sleep(delayT + offset)


    def carSlide(self, speed):
        if self.verbose:
            print("CAR SLIDE:")
        self.setFourMotors(speed, speed, -speed, -speed)

    
    def carMixed(self, v_straight, v_rotate, v_slide):
        if self.verbose:
            print("CAR MIXED")
        self.setFourMotors(
            v_rotate-v_straight+v_slide,
            v_rotate+v_straight+v_slide,
            v_rotate-v_straight-v_slide,
            v_rotate+v_straight-v_slide
        )
    
    def close(self):
        self.bot.close()
        self.bot.exit()


if __name__ == "__main__":
    import time
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)
    mpi_ctrl.carStraight(50)
    time.sleep(0.2)
    
    #mpi_ctrl.carSlide(100)
    #time.sleep(2)
    
    #mpi_ctrl.carRotate(-100)
    #time.sleep(2.104) 

    #rotate takes (speed, [45, 90, 180, or 360], time object)
    #mpi_ctrl.carRotateWithDegree(100, 180, time)

    # mpi_ctrl.carStraight(70)
    # time.sleep(4)

    # mpi_ctrl.carRotateWithDegree(100, 180, time)

    mpi_ctrl.carStop()
    # print("If your program cannot be closed properly, check updated instructions in google doc.")
