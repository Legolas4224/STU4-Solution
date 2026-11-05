import numpy as np
import spaceteams as st
from datetime import datetime, timedelta
from collections import deque

class Rock :
    def __init__(self, obj_matrix, shuttle_pos) :
        self.position = obj_matrix[0] + shuttle_pos # adds to the shuttle's pos relative to Phobos. That way it's an absolute position that can actually be compared.
        self.diameter = obj_matrix[1][-1]
        self.seen = 1
        self.birthday = obj_matrix[2][0]
        self.last_seen = self.birthday
        self.past_positions = deque(maxlen=10) # populate with timestamps so it's easier to calculate velocities
        self.past_positions.append(self.position)

        st.OnScreenLogMessage(f"New object with position {self.position}, diameter {self.diameter}m", "Asteroids", st.Severity.Info)
        #print(f"New object with position {self.position}")
    
    def update(self, obj_matrix, shuttle_pos, time) :
        self.past_positions.append(self.position)
        self.position = obj_matrix[0] + shuttle_pos
        self.last_seen = time
        self.seen += 1
        self.avg_vel(time)

    def avg_vel(self, time) :
        if len(self.past_positions) > 3 :
            self.velocity = (self.past_positions[-1]-self.past_positions[0])/(time-self.birthday)
            st.OnScreenLogMessage(f"Object Velocity Estimate: {self.velocity}", "Asteroids", st.Severity.Info)

class Tracker :
    def __init__(self) :
        self.rocks = []

    def add_rock(self, new_obj, shuttle_loc) :
        if len(self.rocks) != 0 :
            correlation = False
            for rock in self.rocks : # Check all current objects to see if it matches
                delta_diam = abs(rock.diameter - new_obj[1][-1])
                delta_pos = np.linalg.norm(rock.position - (new_obj[0] + shuttle_loc))
                obs_time = new_obj[2][0]
                if delta_diam < 1 and delta_pos < 2 :
                    st.OnScreenLogMessage(f"Object may already have been seen \n d diff: {delta_diam}\n delta pos: {delta_pos}. Not creating new entry", "Asteroids", st.Severity.Info)
                    rock.update(new_obj, shuttle_loc, obs_time)
                    correlation = True
                    continue

            if correlation == False :
                new_rock = Rock(new_obj, shuttle_loc)
                self.rocks.append(new_rock)
                st.OnScreenLogMessage(f"Object looks new. Appending", "Asteroids", st.Severity.Info)
        else : # If the list is empty, add whatever you see!
            new_rock = Rock(new_obj, shuttle_loc)
            self.rocks.append(new_rock)
            st.OnScreenLogMessage(f"Object looks new. Appending", "Asteroids", st.Severity.Info)

    def trash_time(self,time) :
        st.OnScreenLogMessage(f"Current # of  traces: {len(self.rocks)}", "Asteroids", st.Severity.Info)
        self.rocks = [r for r in self.rocks if time - r.last_seen < 5]

    def show(self): 
        pass