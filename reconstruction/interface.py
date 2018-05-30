import sys, os, math
import iotbx.phil
import random,time
from iotbx.option_parser import option_parser
import libtbx.phil.command_line
from cStringIO import StringIO
from libtbx.utils import null_out
from libtbx.utils import Sorry, date_and_time, multi_out


def additional_help_txt(master_params,out=None):
    if out is None:
      out = sys.stdout

    effective_params = master_params.fetch(sources=[])
    params = effective_params.extract()
    new_params =  master_params.format(python_object=params)
    new_params.show(out=out,expert_level=1,attributes_level=1)


def show_help(master_params,help_function,out=None):
    if out is None:
      out = sys.stdout
    if help_function is not None:
      help_function(out)
    print >> out

    additional_help_txt(master_params,out)



def get_input(args, master_params, home_scope, banner, help_function=None, log=None):

  if log is None:
      log = sys.stdout
  
  log2 = sys.stdout
  if (len(args) == 0 or "--help" in args or "--h" in args or "-h" in args):
    if help_function is None:
      print >> log, "No help available yet."
      print >> log2, "No help available yet."
    else:
      show_help(master_params, help_function,log)
      show_help(master_params, help_function,log2)
      return None
  else:
    
    phil_objects = []
    argument_interpreter = libtbx.phil.command_line.argument_interpreter(
      master_phil=master_params,
      home_scope="tipsy")

    for arg in args:
      command_line_params = None
      arg_is_processed = False
      # is it a file?
      if (os.path.isfile(arg)): ## is this a file name?
        # check if it is a phil file
        try:
          command_line_params = iotbx.phil.parse(file_name=arg)
          if command_line_params is not None:
            phil_objects.append(command_line_params)
            arg_is_processed = True
        except KeyboardInterrupt: raise
        except : pass
      else:
        try:
          command_line_params = argument_interpreter.process(arg=arg)
          if command_line_params is not None:
            phil_objects.append(command_line_params)
            arg_is_processed = True
        except KeyboardInterrupt: raise
        except : pass

      if not arg_is_processed:
        print >> log, "##----------------------------------------------##"
        print >> log, "## Unknown file or keyword:", arg
        print >> log, "##----------------------------------------------##"
        print >> log

        print >> log2, "##----------------------------------------------##"
        print >> log2, "## Unknown file or keyword:", arg
        print >> log2, "##----------------------------------------------##"
        print >> log2

        raise Sorry("Unknown file or keyword: %s" % arg)

    effective_params = master_params.fetch(sources=phil_objects)
    params = effective_params.extract()
    new_params =  master_params.format(python_object=params)

    return params


def tst_help(out=None):
  if out is None:
    out = sys.stdout
  print >> out, "This should be a help function"

def tst():
  master_params = iotbx.phil.parse("""\
    test{
      data = None
      .type=path
      .help = "q Intensity Sigma"
      .multiple=True

      d_max = None
      .type=float
      .help="Maximum distance in particle"

      scan = False
      .help="When True, a dmax scan will be performed"
      .type=bool
     }""")
  banner = "------- This is a test -------"
  params = get_input( ["data=data.dat", "d_max=1", "scan=True"], master_params, "whatever", banner, tst_help)
  params = get_input( [], master_params, "whatever", banner, tst_help)

if __name__ == "__main__":
  tst()
