import sys
import math
import time


class ProgressBar:
    def __init__(self, iterations, label, num_of_signs=25):
        # initialize progress bar object
        self.__iterations = iterations
        self.__label = label
        self.__passed = 0
        self.__num_of_signs = num_of_signs
        self.__start_time = time.time()

    def update(self):
        # increment progress bar value, calculate new percentages and number of "=" signs
        self.__passed += 1
        per = int(self.__passed / self.__iterations * 100)
        num_of_signs = math.floor(per * self.__num_of_signs / 100)
        # print progress bar value
        sys.stdout.write("\r" + self.__label + " - " + "%d" % per
                         + "% [" + ('=' * num_of_signs) + (' ' * (self.__num_of_signs - num_of_signs)) + "]")
        sys.stdout.flush()

    def __del__(self):
        # print finish message
        sys.stdout.write("\r" + self.__label + " - " + "%d" % 100 + "%, duration - " +
                         str(time.time() - self.__start_time) + " seconds.\n")
        sys.stdout.flush()
