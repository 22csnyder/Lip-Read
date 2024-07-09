# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:08:16 2015

@author: christopher
"""

import time
import heapq
from sched import scheduler
import cv2


class my_scheduler(scheduler):
    '''
    This class is a light wrap over the original scheduler
    The advantage is that it will warn you if the method takes longer than the time alloted
    The disadvantage is that it only works for 1 scheduled method per time
    '''
    
    delay_was_called=False#should be called once before each action called
    
    def run(self): #I am rewriting this method to warn when method takes too long
        
        # localize variable access to minimize overhead
        # and to improve thread safety
        q = self._queue
        delayfunc = self.delayfunc
        timefunc = self.timefunc
        pop = heapq.heappop
        while q:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                break
            time, priority, action, argument = checked_event = q[0]
            now = timefunc()
            if now < time:
                delayfunc(time - now)#time.sleep is called
                self.delay_was_called=True
            else:
                if self.delay_was_called is False:
                    print 'method took longer than period allotted'
                self.delay_was_called=False #reset
                event = pop(q)
                # Verify that the event was not removed or altered
                # by another thread after we last looked at q[0].
                if event is checked_event:
                    action(*argument)
                    delayfunc(0)   # Let other threads run
                else:
                    heapq.heappush(q, event)

    def new_timed_call(self,calls_per_second, callback, *args, **kw):
        period = 1.0 / calls_per_second
        def reload():
            callback(*args, **kw)
            scheduler.enter(period, 0, reload, ())#enter is relative time
        scheduler.enter(period, 0, reload, ())


#scheduler = scheduler(time.time, time.sleep)
scheduler = my_scheduler(time.time, time.sleep)




#### example code ####

def p(c):
    "print the specified character"
    print c,

def q(c):
    time.sleep(2)
    print c



if __name__ == '__main__':
    scheduler.new_timed_call(3, p, '3p')  # print '3' three times per second
    #scheduler.new_timed_call(4, q, '4q')  # print '3' three times per second
    scheduler.new_timed_call(.1, q, '.1q')  # print '.1' .1 times per second
    #scheduler.new_timed_call(9, q, '9')  # print '9' nine times per second
    
    
    
    
    
    
    scheduler.run()
    
    
    



