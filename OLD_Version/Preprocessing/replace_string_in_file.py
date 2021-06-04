#!/home/leto/miniconda3/bin/python 

##################################################
## Content: This Script replaces a string within 
# a file with another string

## Author: Stanislav Sys

## Last Update: 27.10.20

## Usage: This is a command line scripts. U have
# specifiy the paramters:
# string_to_replace
# string_replacement
# input_filename
# output_filename

# If you are on Linux you can also use sed, which is made for this stuff:
# example:
# replace /home/noobgod/Desktop/collembola_ai/train/ with noob/ in 0.xml 
# sed -e "s/\/home\/noobgod\/Desktop\/collembola_ai\/train\//noob\//" 0.xml > new_noob.xml


##################################################

#imports
import os                               # I/O
import sys                              # Args



# Arguments
# get the number of input arguments
number_args = len(sys.argv)


# Functions
# opens a file and replaces a string in a file
def replace_string_in_file(string_to_replace, string_replacement, input_filename, output_filename):
    print("replacing {} with {} in {}".format(string_to_replace,string_replacement, input_filename))
    #open file
    my_file = open(input_filename, "rt")
    #read file in a variable
    my_data = my_file.read()
    #replace string in read in file
    my_data = my_data.replace(string_to_replace, string_replacement)
    #close file
    my_file.close()
    #open file in write mode
    my_file = open(output_filename, "wt")
    #write variable to file
    my_file.write(my_data)
    #close file
    my_file.close()

    


# check the number of args
if not 3 <= number_args <= 5:
    print("usage:  string_to_replace string_replacement input_filename outputfilename")
else:
    # call replace funtion 
    replace_string_in_file(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

    
# example:  python replace_string_in_file.py Hello GoodBye Hello_world.py GoodBye_world.py
