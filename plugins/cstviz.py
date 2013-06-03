'''
---CSTViz Magnetic shielding tensor visualization plugin for PyMOL---

Author: Martin Babinsky <martbab@chemi.muni.cz>
Date: March 2012

------

TODO: 
    1.) Make the plugin load the molecular geometry along the CST data directly 
        from the Gaussian ouput file.

    2.) enable the coloring of pseudoatoms by their corresponding components
        currently they are colored only as red/green/blue

    3.) make the plugin store and retrieve user-defined settings in a config
        file

    4.) rewrite the 'set_defaults' method so it works properly

"""
This plugin enables the users to visualize the results of quantum-chemical 
calculation of chemical shielding tensors in PyMOL. Currently only 
visualization of results obtained from Gaussian(R) calculation is supported.
The plugin is tested with Pymol 1.4.1 and Gaussian ver 03 and 09 programs.

To use CSTViz, you will need the output file from Gaussian calculation ran
with the keyword 

"NMR(PrintEigenVectors)" 

in the route file. Additionaly, you will need some molecular coordinate file 
to load, ideally with the same coordinates as the Gaussian input geometry. 
You can then use the GUI found under the 'Plugins' menu, or the command line 
interface to visualize data.


The GUI part of this plugin is partially based on the APBS Tools plugin by
Michael G. Lerner.

KNOWN ISSUES:
    1.) The 'Restore defaults' button does not work as expected. Probably
        the method 'CSTVizGUI.set_defaults' does not work as expected


This software is under development and I'm quite new to programming, so if you
find some bug, weird piece of code or just want to give some constructive 
advice, then by all means let me know.
"""
'''

##############################################################################
# 
# Lower level implementation of classes facilitating the file I/O and
# representation of NMR chemical shielding data
##############################################################################
from numpy import array, sum, dot
from numpy.linalg import norm
from pymol import cmd, stored, CmdException
from pymol.cgo import *
import inspect
from copy import deepcopy


# debugging symbol to make developer's life a bit easier
DEBUG = True

def print_debug_info(action, *args):
    '''print some useful debugging information using \"inspect\" module.
    can print the \"action\" performed on \"*args\"'''
    debug_stack = inspect.stack()

    debug_prefix = "::DEBUG<%6d>:: " % debug_stack[1][2]
    debug_msg = debug_prefix + "method \"%s\" called by \"%s\"\n" % \
        (debug_stack[1][3], debug_stack[2][3])

    if action is not None:
        debug_msg += debug_prefix + " " * 4 + action + " "

    if len(args) != 0:
        debug_msg += repr(args) 

    print debug_msg


class SigmaTensor():
    '''
    Class for storing and manipulating the NMR shielding tensor.
    Facilitates the setting of eigenvalues/eigenvectors, translating the 
    eigenvectors (they are centered at origin of coordinate system by default)
    and also setup of parameters for rendering CST as CGO in PyMOL
    '''

    def __init__(self, 
        nucleus="", 
        index = 0, 
        sigma11 = 0.0, 
        sigma22 = 0.0, 
        sigma33 = 0.0,
        origin = [0.0, 0.0, 0.0],
        eig_vec11 = [0.0, 0.0, 0.0], 
        eig_vec22 = [0.0, 0.0, 0.0], 
        eig_vec33 = [0.0, 0.0, 0.0]
    ):

        self.sigmas = array([sigma11, sigma22, sigma33])

        self.eigenvectors = array([eig_vec11, eig_vec22, eig_vec33])

        self.origin = array(origin)

        self.nucleus = nucleus
        self.index = index 
        self.cgo_width = 0.02
        self.cgo_rel_widths = [ 1.0, 1.0, 1.0 ]
        self.cgo_rel_lengths = [ 1.0, 1.0, 1.0 ]
        self.cgo_colors = [[ 1.0, 0.0, 0.0 ],
                          [ 0.0, 1.0, 0.0 ],
                          [ 0.0, 0.0, 1.0 ]]

        self.draw_pseudo = True
        self.show_pseudo = True
        self.cgo = []
        self.cgo_name = self.nucleus + repr(self.index)
        self.pseudo_name = self.cgo_name + "_comp"

    def __str__(self):
        '''
        prints out a short summary about the tensor parameters as string
        '''
        message = "Nucleus: "
        message += self.nucleus
        message += "\n"

        message += "Eigenvalues: " + repr(self.sigmas) + "\n"
        message += "Isotropic shielding " + repr(self.sigmas.mean()) + "\n"
        message += "Eigenvectors: " + repr(self.eigenvectors) + "\n"
        message += "Tensor origin: " + repr(self.origin) + "\n"
        message += "CGO object width: " + repr(self.cgo_width) + "\n"
        message += "Relative widths: " + repr(self.cgo_rel_widths) + "\n"
        message += "Colors: " + repr(self.cgo_colors) + "\n"
        message += "Draw pseudoatoms: " + repr(self.draw_pseudo) + "\n"
        message += "Show pseudoatoms as spheres: " + repr(self.show_pseudo)

        return message

    def set_cgo_params(self, width=0.02, 
        rel_widths = [ 1.0, 1.0, 1.0 ], 
        rel_lengths = [ 1.0, 1.0, 1.0 ],
        colors = [[1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ]],
        draw_pseudo = True,
        show_pseudo = True
    ):
        '''
        Sets some parameters for the 3-D rendering of shielding tensor
        as a collection of CGOs. Sets up the colors of individual 
        components, overall width of CGOs and relative widths of 
        individual components. Also controls the presentation and drawing 
        of pseudoatoms at the tips of CGO arrows
        '''

        self.cgo_width = width
        self.cgo_rel_widths = rel_widths
        self.cgo_colors = colors
        self.draw_pseudo = draw_pseudo
        self.show_pseudo = show_pseudo
        self.cgo_rel_lengths = cgo_rel_lengths

    def set_cgo_params_gui(self, 
        settings, 
    ):
        '''
        Sets some parameters for the 3-D rendering of shielding tensor
        as a collection of CGOs. Sets up the colors of individual 
        components, overall width of CGOs and relative widths of 
        individual components. Also controls the presentation and drawing 
        of pseudoatoms at the tips of CGO arrows. This method is called from
        the GUI layer (with the dictionary of Tkinter variables controlling
        the appearance as argument).
        '''

        self.cgo_width = settings['cgo_width_var'].get()
        self.cgo_rel_widths = [
            settings['rel_width_11_var'].get(),
            settings['rel_width_22_var'].get(),
            settings['rel_width_33_var'].get()
        ]

        self.cgo_colors = [
            settings['color11'].get_color(),
            settings['color22'].get_color(),
            settings['color33'].get_color()
        ]

        self.draw_pseudo = settings['draw_pseudo_var'].get()
        self.show_pseudo = settings['draw_pseudo_var'].get()
        self.cgo_rel_lengths = [
            settings['rel_length_11_var'].get(),
            settings['rel_length_22_var'].get(),
            settings['rel_length_33_var'].get()
        ]
        if DEBUG:
            print_debug_info("changed settings for item:", 
                "%s%04d" % (self.nucleus, self.index))
            print_debug_info("item settings:", str(self))


    def set_cgo_name(self, name):
        '''sets the name of CGO in PyMOL'''
        self.cgo_name = name + "_CST"
        self.pseudo_name = self.cgo_name + "_comp"

    def delete_cgo(self):
        '''deletes the CGO Object'''
        cmd.delete(self.cgo_name)
        cmd.delete(self.pseudo_name)

    def prepare_cgo(self):
        '''Precalculates the 3D coordinates and colors of CGO representation of 
        shielding tensor'''
        self.cgo = []

        for (n, vec) in enumerate(self.eigenvectors):


            # apply any relative width factor
            cgo_width = self.cgo_rel_widths[n] * self.cgo_width

            # defintion of cone length and end point
            dir_vec = vec * self.cgo_rel_lengths[n]
            dir_vec_length = norm(dir_vec) 
            cone_length = 5 * cgo_width

            # cylinder endpoint definition, made in a way that seems
            # to produce cones that are invariant under scaling
            cylinder_end_point = ((dir_vec / dir_vec_length)\
                * (dir_vec_length - cone_length))

            # end of cones at eigenvector position and its reflection
            # around coordinates of nucleus
            cone1_end = self.origin + dir_vec
            cone2_end = self.origin - dir_vec

            # first define that we draw a cylinder
            self.cgo.append(CYLINDER)

            # the origin of cylinder is at reflected endpoint
            # around nuclear
            self.cgo.extend(self.origin - cylinder_end_point)

            # the end of cylinder is at the endpoint
            self.cgo.extend(self.origin + cylinder_end_point)

            # the width of cylinder
            self.cgo.append(cgo_width)

            # the first and second colors are the same
            self.cgo.extend(self.cgo_colors[n] * 2)

            #then draw a cone
            self.cgo.append(CONE)

            #self.origin of the cone is at the end of cylinder
            self.cgo.extend(self.origin + cylinder_end_point)

            #cone ends at the point of the eigenvector
            self.cgo.extend(cone1_end)

            # diameter equals to 1.618 * width
            self.cgo.append(1.618 * cgo_width)

            # no idea what this number means
            self.cgo.append(0.0)

            # colors, hopefully
            self.cgo.extend(self.cgo_colors[n] * 2)

            # ummm, no idea
            self.cgo.extend([1.0, 1.0])


            #then draw another cone
            self.cgo.append(CONE)

            #origin of the cone is at the origin of cylinder
            self.cgo.extend(self.origin - cylinder_end_point)

            #cone ends at the reflected position of the eigenvector
            self.cgo.extend(cone2_end)

            # diameter equals to 1.618 * width
            self.cgo.append(1.618 * cgo_width)

            # no idea what this number means
            self.cgo.append(0.0)

            # colors, hopefully
            self.cgo.extend(self.cgo_colors[n] * 2)

            # ummm, no idea
            self.cgo.extend([1.0, 1.0])


    def draw_cgo(self):
        '''Do the actual drawing in the PyMol scene and create pseudoatoms if 
        requested'''
        # delete previously drawn CGO
        cmd.delete(self.cgo_name)
        cmd.delete(self.pseudo_name)
        cmd.load_cgo(self.cgo, self.cgo_name)

        if self.draw_pseudo:
            self.pseudo_radius = 1.5 * self.cgo_width
            cmd.pseudoatom(self.pseudo_name, name="11", 
                vdw = self.pseudo_radius,
                b = self.sigmas[0],
                pos = tuple(self.origin + (self.eigenvectors[0] * \
                    self.cgo_rel_lengths[0])),
                color="red"
            )
            cmd.pseudoatom(self.pseudo_name, name="-11", 
                vdw = self.pseudo_radius,
                b = self.sigmas[0],
                pos = tuple(self.origin - (self.eigenvectors[0] * \
                    self.cgo_rel_lengths[0])),
                color="red"
            )
            cmd.pseudoatom(self.pseudo_name, name="22", 
                vdw = self.pseudo_radius,
                b = self.sigmas[1],
                pos = tuple(self.origin + (self.eigenvectors[1] * \
                        self.cgo_rel_lengths[1])), 
                color="green"
            )
            cmd.pseudoatom(self.pseudo_name, name="-22", 
                vdw = self.pseudo_radius,
                b = self.sigmas[1],
                pos = tuple(self.origin - (self.eigenvectors[1] * \
                        self.cgo_rel_lengths[1])), 
                color="green"
            )
            cmd.pseudoatom(self.pseudo_name, name="33", 
                vdw = self.pseudo_radius,
                b = self.sigmas[2],
                pos = tuple(self.origin + (self.eigenvectors[2] * \
                    self.cgo_rel_lengths[2])), 
                color="blue"
            )
            cmd.pseudoatom(self.pseudo_name, name="-33", 
                vdw = self.pseudo_radius,
                b = self.sigmas[2],
                pos = tuple(self.origin - (self.eigenvectors[2] * \
                    self.cgo_rel_lengths[2])), 
                color="blue"
            )
            cmd.show_as("spheres", self.pseudo_name)

            if not self.show_pseudo:
                cmd.hide("everything", self.pseudo_name)



class TensorList():
    '''
    Container of SigmaTensor objects, facilitates reading tensor information 
    from Gaussian output according to selection specs from PyMOL. 
    Also facilitates manipulation of data, like deleting, refreshing of list,
    filtering according to user selection.
    '''
    # format for tensor dictionary keys
    _key_format = "%s%03d"

    def __init__(self):
        # the dictionary of SigmaTensor objects 
        # the keys are in the format 'C10', 'H9' etc.
        self.data = {}
        # the name of the file the data are from
        self.filename = ""
        # PyMOL selection for data filtering
        self.selection = "all"
        # custom name for PyMOL selection
        self.sele_name = "all"
        # shielding type for future use
        self.shielding_type = 'total'

    def insert(self, *args):
        key = ""
        for arg in args:
            if isinstance(arg, SigmaTensor):
                key = self.__class__._key_format % (arg.nucleus, arg.index)
                self.data[key] = arg
            else:
                raise TypeError(
                    "Insertion failed: Invalid type of %s!" % type(arg)
                )

    def set_selection(self, sele = "all", sele_name = ""):
        self.selection = sele
        self.sele_name = sele_name
        if sele_name == "":
            self.sele_name = sele
        for d in self.data:
            self.data[d].delete_cgo()
            self.data[d].set_cgo_name(self.sele_name + "_" + d)

        self.set_sele_atom_dict()

    def set_sele_name(self, sele_name = ""):
        self.sele_name = sele_name

        for d in self.data:
            self.data[d].delete_cgo()
            self.data[d].set_cgo_name(self.sele_name + "_" + d)

    def get_selection(self):
        return self.selection

    def get_sele_name(self):
        return self.sele_name

    def set_sele_atom_dict(self):
        sele_atom_list = cmd.get_model(self.selection).atom

        self.sele_atom_dict = {}
        
        for atom in sele_atom_list:
            atom_key = self.__class__._key_format % (atom.symbol, atom.index)
            self.sele_atom_dict[atom_key] = atom

    def filter_selection(self):
        entries_to_remove = []

        # loop through tensor list and search elements
        # which have the corresponding key in the atom selection
        # when found reset the origin to the position of the selected
        # atom. Otherwise append the element key to the array of elements
        # to be removed
        for d in self.data:
            if d in self.sele_atom_dict:

                self.data[d].origin = array(self.sele_atom_dict[d].coord)
                if DEBUG:
                    print_debug_info("altered following:", d)

                self.data[d].prepare_cgo()
            else:
                entries_to_remove.append(d)
        
        # remove the elements which were not selected
        for key in entries_to_remove:

            if DEBUG:
                print_debug_info("deleted following:", key)
                
            self.data[key].delete_cgo()
            del self.data[key]

    def get_entry_keys(self):
        return sorted(self.data.keys())

    def remove_entries(self, items = []):
        '''deletes the items from the tensor list if they are present.
        item selection is specified as array of strings. If the list is empty,
        then all items are removed'''
        if DEBUG:
            print_debug_info("items to delete:", items)

        if len(items) == 0:
            for d in self.data:
                self.data[d].delete_cgo()

            if DEBUG:
                print_debug_info("deleted all items")

            self.data = {}

        else:
            for i in items:
                if i in self.data:
                    self.data[i].delete_cgo()
                    del self.data[i]

                    if DEBUG:
                        print_debug_info("deleted item", i)

    def redraw_items(self):
        for d in self.data:
            self.data[d].delete_cgo()
            self.data[d].prepare_cgo()
            self.data[d].draw_cgo()

            if DEBUG:
                print_debug_info("redraw item:", d)

    def apply_cgo_settings(self, settings):
        '''apply graphical settings to the items specified by the array
        of keys. When the array is empty the method will alter the CGO 
        settings of all items. The settings are specified as a dictionary
        and passed to the individual tensor objects.
        '''
        for d in self.data:
            self.data[d].set_cgo_params(**Settings)

            if DEBUG:
                print_debug_info("applied CGO settings to item: ", d)


    def apply_cgo_settings_gui(self, settings, items = []):
        '''apply graphical settings to the items specified by the array
        of keys. When the array is empty the method will alter the CGO 
        settings of all items. The settings are specified as a dictionary
        and passed to the individual tensor objects. This is for use in the GUI layer,
        '''

        if len(items) == 0:
            for d in self.data:
                self.data[d].set_cgo_params_gui(settings)
                if DEBUG:
                    print_debug_info("applied CGO settings to item:", d)

        else:
            for i in items:
                if i in self.data:
                    self.data[i].set_cgo_params_gui(settings)
                    self.data[i].prepare_cgo()

                    if DEBUG:
                        print_debug_info("applied CGO settings to item:", i)


class GaussianOutputParser(object):
    '''
    class for parsing Gaussian logfile and loading data to TensorList
    '''
    _section_begin = "SCF GIAO Magnetic shielding tensor (ppm):"
    _section_end = "End of Minotr Frequency-dependent properties file"
    _tensor_begin = "Isotropic ="
    _eigenvalues  = "Eigenvalues:"
    _eigenvectors = "Eigenvectors:"

    def __init__(
        self,
    ):
        pass

    @staticmethod
    def _process_block(
        block
    ):
        result = SigmaTensor()
        result.index = int(block[0][0])
        result.nucleus = block[0][1]

        if (block[4][0] == GaussianOutputParser._eigenvalues
            and
            block[5][0] == GaussianOutputParser._eigenvectors):
            result.sigmas = array(
                map(float, block[4][1:4])
            )
            result.eigenvectors = array(
                [
                    map(float, i[1:4]) for i in block[6:9]
                ]
            )
        else:
            raise IOError(
                "Invalid data format"
            )

        return result

    @staticmethod
    def read(
        filename
    ):
        section_found = False

        result = TensorList()
        tensor_block = None

        with open(filename, 'r') as f:
            for line in f:
                # GaussianParser._line_no += 1
                if GaussianOutputParser._section_begin in line:
                    section_found = True
                    continue

                if section_found:
                    if GaussianOutputParser._section_end in line:
                        break

                    elif GaussianOutputParser._tensor_begin in line: 
                        if tensor_block is None:
                            tensor_block = []
                        else:
                            result.insert(
                                GaussianOutputParser._process_block(
                                    tensor_block
                                )
                            )
                            tensor_block = []

                    tensor_block.append(
                        line.strip().split()
                    )

        return result


##############################################################################
# GUI layer implementation, currently a mess, could use some refactoring
# (c) 2012 Martin Babinsky
##############################################################################
import Pmw
import Tkinter
import tkFileDialog
import tkMessageBox
import tkColorChooser
from os.path import basename
import re

class CSTVizGUI:
    def __init__(self, app):
        self.file_list = {}
        self.parent = app.root
        self.redraw_text = 'Redraw/refresh'
        self.exit_text =  'Exit'

        self.draw_pseudo_var = Tkinter.BooleanVar()
        self.show_pseudo_var = Tkinter.BooleanVar()
        self.cgo_width_var = Tkinter.DoubleVar()

        self.rel_width_11_var = Tkinter.DoubleVar()
        self.rel_width_22_var = Tkinter.DoubleVar()
        self.rel_width_33_var = Tkinter.DoubleVar()

        self.rel_length_11_var = Tkinter.DoubleVar()
        self.rel_length_22_var = Tkinter.DoubleVar()
        self.rel_length_33_var = Tkinter.DoubleVar()

        self.color11 = CGOColor([1, 0, 0])
        self.color22 = CGOColor([0, 1, 0])
        self.color33 = CGOColor([0, 0, 1])

        self.settings_dict = {
            'draw_pseudo_var' : self.draw_pseudo_var,
            'show_pseudo_var' : self.show_pseudo_var,
            'cgo_width_var' : self.cgo_width_var,
            'rel_width_11_var' : self.rel_width_11_var,
            'rel_width_22_var' : self.rel_width_22_var,
            'rel_width_33_var' : self.rel_width_33_var,
            'rel_length_11_var' : self.rel_length_11_var,
            'rel_length_22_var' : self.rel_length_22_var,
            'rel_length_33_var' : self.rel_length_33_var,
            'color11' : self.color11,
            'color22' : self.color22,
            'color33' : self.color33
        }

        self.selected_file = ""

        self.selected_items = []

        #set the default values of variables
        self.set_defaults()


        #####################################################################
        # The main window configuration
        # with label on top and bottom buttons
        #
        #####################################################################
        self.app_title = 'NMR Shielding Tensors visualization plugin'
        self.top_label_text = "(c) 2012 Martin Babinsky: \
                martbab (at) chemi (dot) muni (dot) cz"
        self.top = Pmw.Dialog(self.parent,
            buttons = (self.redraw_text, self.exit_text),
            command = self.execute
        )
        self.top.title(self.app_title)
        self.top.withdraw()

        self.top.component('hull').geometry('600x400')
        Pmw.setbusycursorattributes(self.top.component('hull'))
        self.top.protocol("WM_DELETE_WINDOW", self.exit)
        self.topLabel = Tkinter.Label(self.top.interior(), 
            text = self.top_label_text
        )
        self.topLabel.grid(row = 0, 
            column = 0, 
            sticky="we"
        )
        #####################################################################
        # the notebook class to organize GUI into two groups
        # 1) List of loaded files/tensors, adding/deleting items, configuring
        #    individual items
        # 2) Global settings
        #####################################################################
        self.note_book = Pmw.NoteBook(self.top.interior())
        self.note_book.grid(
            column = 0,
            row = 1,
            sticky = "nsew"
        )

        self.top.interior().grid_columnconfigure(0,
            weight = 1
        )
        self.top.interior().grid_rowconfigure(1,
            weight = 1,
        )
        self.cst_manager = self.note_book.add('Data lists')
        self.global_settings = self.note_book.add('Global settings')

        #####################################################################
        # CST Manager controls loading/deleting of new items and individual
        # configuration
        #
        #####################################################################
        self.cst_manager.rowconfigure(0,
            weight = 1
        )
        self.cst_manager.columnconfigure(0,
            weight = 1
        )

        self.cst_manager_handler(self.cst_manager)

        #####################################################################
        # The general settings section controls the default global settings
        # these can be overriden by individual settings
        #####################################################################
        self.global_settings_handler = SettingsWindow(self.global_settings,
            apply_command = self.apply_global_settings,
            set_defaults_command = self.set_defaults,
            **self.settings_dict
        )

        self.show_app_modal()

    def cst_manager_handler(self, parent):
        self.files_items_window(parent)

    def files_items_window(self, parent):
        self.file_list_frame = Tkinter.LabelFrame(parent,
            text = "List of loaded files"
        )
        self.file_list_frame.grid(row = 0,
            column = 0,
            sticky = "nsew"
        )
        self.file_list_frame.columnconfigure(0, 
            weight = 1,
        )
        self.file_list_frame.rowconfigure(0, 
            weight = 1,
        )

        self.file_contents_frame = Tkinter.LabelFrame(parent,
            text = "file contents"
        )

        self.file_contents_frame.grid(row = 0,
            column = 1,
            sticky = "nsew"
        )
        self.file_contents_frame.columnconfigure(0, 
            weight = 1,
        )
        self.file_contents_frame.rowconfigure(0, 
            weight = 1,
        )
        self.file_list_component(self.file_list_frame)
        self.file_contents_listbox_component(self.file_contents_frame)
        
    def file_list_component(self, parent):
        self.file_listbox = Pmw.ScrolledListBox(parent,
            listbox_height = 8,
            selectioncommand = self.select_file,
        )

        self.file_listbox.grid(
            row = 0,
            column = 0,
            columnspan = 1,
            sticky = 'nsew'
        )

        self.file_list_button_frame = Tkinter.Frame(parent)
        self.file_list_button_frame.grid(
            row = 0,
            column = 1,
            sticky = 'nsew'
        )

        self.file_list_button_group(self.file_list_button_frame)
        self.selection_filter_frame = Tkinter.LabelFrame(
            parent,
            text = "data filtering",
        )
        self.selection_filter_frame.grid(
            row = 4,
            column = 0, 
            columnspan = 2,
            sticky = 'nsew'
        )

        self.selection_filter_group(self.selection_filter_frame)
        
        ###########################
        # menu for file list
        ###########################

        self.file_listbox_menu = Tkinter.Menu( 
            parent,
            tearoff = 0,
        )

        self.file_listbox_menu.add_command( 
            label = "Add file(s)",
            command = self.open_file,
        )
        self.file_listbox_menu.add_command( 
            label = "Reload File",
            command = self.reload_file,
        )
        self.file_listbox_menu.add_command( 
            label = "Limit to selection",
            command = self.filter_pymol_sele
        )
        self.file_listbox_menu.add_command( 
            label = "Settings",
            command = lambda: self.local_settings_window(parent),
        )
        self.file_listbox_menu.add_command( 
            label = "Remove file",
            command = self.remove_file,
        )
        ###########################
        self.file_listbox.component("listbox").bind(
            "<Button-3>", 
            self.file_menu
        )

    def file_list_button_group(self, parent):
        self.load_file_button = Tkinter.Button(
            parent,
            text = "Open file(s)",
            command = self.open_file,
        )
        self.load_file_button.grid(
            row = 0,
            column = 0,
            sticky='we'
        )

        self.reload_file_button = Tkinter.Button(
            parent,
            text = "Reload file",
            command = self.reload_file,
        )
        self.reload_file_button.grid(
            row = 2,
            column = 0,
            sticky = 'we'
        )
        self.remove_file_button = Tkinter.Button(
            parent,
            text = "Remove file",
            command = self.remove_file,
        )

        self.remove_file_button.grid(
            row = 3,
            column = 0,
            sticky = 'we'
        )

        self.remove_all_files_button = Tkinter.Button(
            parent,
            text = "Remove all files",
        )

        self.remove_all_files_button.grid(
            row = 4,
            column = 0,
            sticky = 'we'
        )

    def selection_filter_group(self, parent):
        self.selection_entry = Pmw.EntryField( 
            parent,
            command = self.filter_pymol_sele,
            label_text = "selection",
            labelpos = 'nw',
            value = 'all',
        )
        self.selection_entry.grid(
            row = 0,
            column = 0,
            sticky = 'we'
        )
        self.selection_name_entry = Pmw.EntryField( 
            parent,
            command = self.rename_pymol_sele,
            label_text = "name for selection",
            labelpos = 'nw',
            value = 'sele1',
        )
        self.selection_name_entry.grid(
            row = 1,
            column = 0,
            sticky = 'we'
        )

        self.get_selection_button = Tkinter.Button( 
            parent,
            text = 'get PyMOL selection',
            command = self.choose_pymol_sele,
        )
        self.get_selection_button.grid(
            row = 2,
            column = 0,
            sticky = 'we'
        )
        self.filter_selection_button = Tkinter.Button( 
            parent,
            text = 'Filter data',
            command = self.filter_pymol_sele,
        )
        self.filter_selection_button.grid(
            row = 3,
            column = 0,
            sticky = 'we'
        )
    def file_contents_listbox_component(self, parent):
        self.file_contents_listbox = Pmw.ScrolledListBox(self.file_contents_frame,
            listbox_height = 10,
            listbox_selectmode = "extended",
            selectioncommand = self.get_selected_items
        )
        self.file_contents_listbox.grid(
            row = 0,
            column = 0,
            sticky = 'nsew'
        )
        self.file_contents_buttons_frame = Tkinter.Frame(parent)
        self.file_contents_buttons_frame.grid(
            row = 0,
            column = 1,
            sticky = 'nsew'
        )
        self.file_contents_buttons_group(self.file_contents_buttons_frame)

        ###########################
        # menu for CSTs
        ###########################
        self.datafile_contents_menu = Tkinter.Menu(parent, tearoff = 0)
        self.datafile_contents_menu.add_command(label = "Settings", 
            command = lambda: self.local_settings_window(parent),
        )
        self.datafile_contents_menu.add_command(label = "Remove",
            command = self.remove_selected_items,
        )
        self.datafile_contents_menu.add_command(label = "Clear List",
            command = self.clear_all_items,
        )
        ###########################
        self.file_contents_listbox.component("listbox").bind("<Button-3>", 
            self.cst_menu
        )

    def file_contents_buttons_group(self, parent):
        self.graphics_settings_button = Tkinter.Button( 
            parent,
            text = "Edit",
            command = lambda: self.local_settings_window(parent),
        )

        self.graphics_settings_button.grid(
            row = 0,
            column = 0,
            sticky = 'ew'
        )
        self.show_info_button = Tkinter.Button( 
            parent,
            text = "Show info",
        )
        self.show_info_button.grid(
            row = 1,
            column = 0,
            sticky = 'ew',
        )
        self.reload_list_button = Tkinter.Button( 
            parent,
            text = "Reload list",
            command = self.reload_file,
        )
        self.reload_list_button.grid(
            row = 2,
            column = 0,
            sticky = 'ew',
        )
        self.remove_items_button = Tkinter.Button( 
            parent,
            text = "Remove",
            command = self.remove_selected_items,
        )
        self.remove_items_button.grid(
            row = 3,
            column = 0,
            sticky = 'ew'
        )
        self.remove_all_button = Tkinter.Button( 
            parent,
            text = "Remove all",
            command = self.clear_all_items,
        )
        self.remove_all_button.grid(
            row = 4,
            column = 0,
            sticky = 'ew',
        )


    def cst_menu(self, event):
        self.datafile_contents_menu.tk_popup(event.x_root, event.y_root)
        self.selected_items = self.file_contents_listbox.getcurselection()

        if DEBUG:
            print "::DEBUG::class \"%s\", method \"%s\": menu called at %d %d" % \
                    (self.__name__, self.cst_menu.__name__, event.x_root, event.y_root)
            print "::DEBUG::\tselected items list: %s" % (repr(self.selected_items))

    def file_menu(self, event):
        self.file_listbox_menu.tk_popup(event.x_root, event.y_root)

        try:
            self.selected_items = self.file_list[self.selected_file].get_entry_keys()
        except KeyError:
            pass

    def select_file(self):
        try:
            self.selected_file = self.file_listbox.getcurselection()[0]
            self.update_file_contents_listbox()
            self.selected_items = self.file_list[self.selected_file].get_entry_keys()
            self.selection_entry.setvalue(
                self.file_list[self.selected_file].get_selection()
            )
            self.selection_name_entry.setvalue(
                self.file_list[self.selected_file].get_sele_name()
            )
        except IndexError, KeyError:
            pass

    def open_file(self):
        files = tkFileDialog.askopenfilenames(
            parent = self.file_list_frame,
            title = "Choose a Gaussian output"
        )
        file_list = []
        
        # the following lines address the issue #5712 in Tkinter module
        # on Windows OS. The askopenfilenames method returns a string instead
        # of a tuple, so we have to split it to several filenames manually
        if type(files) is not tuple:
            file_list = self.parent.tk.splitlist(files)
        else:
            file_list = files
            
        for f in file_list:
            if not f in self.file_list:
                self.file_list[f] = self.read_file(
                    f
                )
                #print basename(f)
                #self.file_list[f].read(f)
                self.file_listbox.insert("end", f)

                self.file_list[f].apply_cgo_settings_gui(
                    self.settings_dict
                )

    def read_file(
        self, 
        filename, 
        file_type = 'gaussian'
    ):
        result = None

        if file_type == 'gaussian':
            result = GaussianOutputParser.read(filename)
        else:
            raise NotImplementedError(
                "This file format is not (yet) supported by CSTViz"
            )
            
        return result
        

    def reload_file(self):
        try:
            self.clear_all_items()
            self.file_list[self.selected_file] = self.read_file(
                self.selected_file
            )
            self.file_list[self.selected_file].apply_cgo_settings_gui(
                self.settings_dict
            )

            self.update_file_contents_listbox()
        except KeyError, IndexError:
            pass

    def remove_file(self):
        self.clear_all_items()
        try:
            del self.file_list[self.selected_file]
            self.file_listbox.delete("active")
        except KeyError:
            pass

    def get_selected_items(self):
        self.selected_items = self.file_contents_listbox.getcurselection()

    def choose_pymol_sele(self):
        selections = cmd.get_names('all')

        self.filter_sele_combobox = Pmw.ComboBoxDialog(self.top.interior(),
            title = 'filter data by molecule/selection',
            label_text = 'choose PyMOL object',
            combobox_labelpos = 'n',
            buttons = ('Accept', 'Cancel'),
            scrolledlist_items = tuple(selections),
        )
        self.filter_sele_combobox.withdraw()
        result = self.filter_sele_combobox.activate()
        selection = self.filter_sele_combobox.get()

        if selection != "":
            self.selection_entry.setvalue(selection)

    def filter_pymol_sele(self):
        try:
            selection = self.selection_entry.getvalue()
            selection_name = self.selection_name_entry.getvalue()

            self.file_list[self.selected_file].set_selection(sele = selection,
                sele_name = selection_name
            )
            self.file_list[self.selected_file].filter_selection()
            self.update_file_contents_listbox()
        except CmdException, KeyError:
            pass

    def get_selection(self):
        try:
            return self.file_list[self.selected_file].get_selection()
        except KeyError:
            return 'all'

    def get_sele_name(self):
        try:
            return self.file_list[self.selected_file].get_sele_name()
        except KeyError:
            return 'all'

    def rename_pymol_sele(self):
        try:
            selectionName = self.selection_name_entry.getvalue()
            self.file_list[self.selected_file].set_sele_name(selectionName)
        except CmdError:
            pass
        except KeyError:
            pass

        self.update_file_contents_listbox()


    def update_file_contents_listbox(self):

        try:
            self.file_contents_listbox.setlist(
                self.file_list[self.selected_file].get_entry_keys()
            )
        except KeyError,IndexError:
            self.file_contents_listbox.clear()

    def local_settings_window(self, parent):
        self.window_parent = Tkinter.Toplevel(parent)
        self.local_settings = SettingsWindow(self.window_parent,
            apply_command = lambda: self.apply_local_settings(self.window_parent),
            set_defaults_command = self.set_defaults,
            **self.settings_dict
        )


    def remove_selected_items(self):
        self.file_list[self.selected_file].remove_entries(self.selected_items)
        # for sel in self.selected_items:
        #    self.file_list[self.selected_file][sel].delete_cgo()
        #    del self.file_list[self.selected_file][sel]

        self.update_file_contents_listbox()



    def set_defaults(self):
        self.draw_pseudo_var.set(True)
        self.draw_pseudo_var.set(True)
        self.cgo_width_var.set(0.05)

        self.rel_width_11_var.set(1.0)
        self.rel_width_22_var.set(1.0)
        self.rel_width_33_var.set(1.0)

        self.rel_length_11_var.set(1.0)
        self.rel_length_22_var.set(1.0)
        self.rel_length_33_var.set(1.0)

        self.color11.set_color([1.0, 0.0, 0.0])
        self.color22.set_color([0.0, 1.0, 0.0])
        self.color33.set_color([0.0, 0.0, 1.0])

    def apply_local_settings(self, parent):
        self.file_list[self.selected_file].apply_cgo_settings_gui(
            self.settings_dict,
            items = self.selected_items
        )
        parent.destroy()

    def apply_global_settings(self):
        for filename in self.file_list:
            self.file_list[filename].apply_cgo_settings_gui(
                self.settings_dict
            )

    def clear_all_items(self):
        self.file_contents_listbox.clear()

        try:
            self.file_list[self.selected_file].remove_entries()
        except KeyError:
            pass

    def redraw_items(self):
        for filename in self.file_list:
            self.file_list[filename].redraw_items()

    def exit(self, event = None):
        self.top.destroy()


    def execute(self, event = None):
        if event == self.redraw_text:
            self.redraw_items()
        else:
            self.exit()

    def show_app_modal(self):
        self.top.show()

#############################################################################
#
# some parts of GUI written as separate classes to improve readability a bit
#############################################################################
class SettingsWindow:
    '''
    widget for the settings, etiher local or global
    '''
    def __init__(
        self, 
        parent, 
        **kwargs
    ):
        self.parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "SettingsWindow", arg, type(getattr(self, arg))

        self.parent.rowconfigure(0,
            weight = 1
        )
        self.parent.columnconfigure(0,
            weight = 1
        )
        self.pseudo_settings = Tkinter.LabelFrame(self.parent,
            text = "Pseudoatoms",
        )
        self.pseudo_settings.grid(row = 0,
            column = 0,
            sticky = 'new',
            columnspan = 2,
        )
        self.pseudo_settings.columnconfigure(0,
            weight = 1,
        )
        self.graphics_settings = Tkinter.LabelFrame(self.parent,
            text = "Graphical settings",
        )
        self.graphics_settings.grid(row = 1,
            column = 0,
            sticky = 'new',
            columnspan = 2,
        )
        self.graphics_settings.columnconfigure(1,
            weight = 1,
        )
        self.pseudo_settings_handler(self.pseudo_settings)
        self.graphics_settings_handler(self.graphics_settings)
        self.apply_button = Tkinter.Button(self.parent,
            text = 'Apply settings',
            command = self.apply_command,
        )
        self.apply_button.grid(row = 2,
            column = 0,
            sticky = 'we'
        )
        self.global_defaults_button = Tkinter.Button(self.parent,
            text = 'Restore defaults',
            command = self.set_defaults_command,
        )
        self.global_defaults_button.grid(row = 2,
            column = 1,
            sticky = 'we'
        )

    def pseudo_settings_handler(self, parent):
        self.draw_pseudo_checkbutton = Tkinter.Checkbutton(parent,
           text = 'Draw pseudoatoms along CGO arrows',
           variable = self.draw_pseudo_var,
        )
        self.draw_pseudo_checkbutton.grid(
            row = 0,
            column = 0,
            sticky = 'w'
        )

        self.show_pseudo_checkbutton = Tkinter.Checkbutton(parent,
            text = 'Show pseudoatoms',
            variable = self.show_pseudo_var,
        )
        self.show_pseudo_checkbutton.grid(
            row = 1,
            column = 0,
            sticky = 'w'
        )

    def graphics_settings_handler(self, parent):
        self.absolute_width_spinbox = Tkinter.Spinbox(parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.01,
            width = 5,
            textvariable = self.cgo_width_var,
        )

        self.absolute_width_spinbox.grid(row = 0,
            column = 0,
            sticky = 'w',
        )
        
        self.absolute_width_label = Tkinter.Label(parent,
            text = 'Absolute width of CGO arrows',
        )

        self.absolute_width_label.grid(row = 0,
            column = 1,
            sticky = 'w',
        )
        self.rel_width_group = Tkinter.LabelFrame(parent,
            text = 'Relative widths of tensor components',
            borderwidth = 0,
        )
        self.rel_width_group.grid(row = 1,
            column = 0, 
            sticky = 'we',
            columnspan = 2
        )
        self.rel_widths_spinbox_group = RelWidthsSpinBoxes(self.rel_width_group,
            rel_width_11_var = self.rel_width_11_var,
            rel_width_22_var = self.rel_width_22_var,
            rel_width_33_var = self.rel_width_33_var
        )
        self.rel_lengths_group = Tkinter.LabelFrame(parent,
            text = 'Relative lengths of tensor components',
            borderwidth = 0,
        )
        self.rel_lengths_group.grid(row = 2,
            column = 0,
            sticky = 'we',
            columnspan = 2,
        )

        self.rel_lengths_spinbox_group = RelLenSpinBoxes(self.rel_lengths_group,
            rel_length_11_var = self.rel_length_11_var,
            rel_length_22_var = self.rel_length_22_var,
            rel_length_33_var = self.rel_length_33_var
        )

        self.color_group = Tkinter.LabelFrame(parent,
            text = "Color of tensor components",
            borderwidth = 0,
        )
        self.color_group.grid(row = 1,
            column = 2,
            sticky = 'nw'
        )
        self.component_color_group = ComponentColorButtons(self.color_group,
                color11 = self.color11,
                color22 = self.color22,
                color33 = self.color33
        )

class RelWidthsSpinBoxes:
    '''
    Widget controlling the relative width of CGO components
    '''
    def __init__(self, parent, **kwargs):
        self.parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "RelWidthsSpinBoxes", arg, type(getattr(self, arg))

        self.rel_width_11_spinbox = Tkinter.Spinbox(self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_width_11_var,
        )
        self.rel_width_11_spinbox.grid(row = 0,
            column = 0,
            sticky = 'w'
        )

        self.rel_width_22_spinbox = Tkinter.Spinbox(self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_width_22_var,
        )
        self.rel_width_22_spinbox.grid(row = 1,
            column = 0,
            sticky = 'w'
        )

        self.rel_width_33_spinbox = Tkinter.Spinbox(self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_width_33_var,
        )
        self.rel_width_33_spinbox.grid(row = 2,
            column = 0,
            sticky = 'w'
        )

        self.rel_width_11_label = Tkinter.Label(self.parent,
            text = '11',
        )

        self.rel_width_11_label.grid(row = 0,
            column = 1,
            sticky = 'w'
        )

        self.rel_width_22_label = Tkinter.Label(self.parent,
            text = '22',
        )

        self.rel_width_22_label.grid(row = 1,
            column = 1,
            sticky = 'w'
        )

        self.rel_width_33_label = Tkinter.Label(self.parent,
            text = '33',
        )

        self.rel_width_33_label.grid(row = 2,
            column = 1,
            sticky = 'w'
        )

class RelLenSpinBoxes:
    '''
    widget controlling the relative length of CGO components
    '''
    def __init__(self, parent, **kwargs):

        self.parent = parent
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "RelLenSpinBoxes: ", arg, type(getattr(self, arg))

        self.rel_length_11_spinbox = Tkinter.Spinbox(
            self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_length_11_var,
        )
        self.rel_length_11_spinbox.grid(
            row = 0,
            column = 0,
            sticky = 'w'
        )


        self.rel_length_22_spinbox = Tkinter.Spinbox(
            self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_length_22_var,
        )
        self.rel_length_22_spinbox.grid(row = 1,
            column = 0,
            sticky = 'w'
        )

        self.rel_length_33_spinbox = Tkinter.Spinbox(
            self.parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.rel_length_33_var,
        )
        self.rel_length_33_spinbox.grid(row = 2,
            column = 0,
            sticky = 'w'
        )

        self.rel_length_11_label = Tkinter.Label(
            self.parent,
            text = '11',
        )

        self.rel_length_11_label.grid(
            row = 0,
            column = 1,
            sticky = 'w'
        )

        self.rel_length_22_label = Tkinter.Label(
            self.parent,
            text = '22',
        )

        self.rel_length_22_label.grid(
            row = 1,
            column = 1,
            sticky = 'w'
        )

        self.rel_length_33_label = Tkinter.Label(
            self.parent,
            text = '33',
        )

        self.rel_length_33_label.grid(
            row = 2,
            column = 1,
            sticky = 'w'
        )

class ComponentColorButtons:
    '''
    widget controlling the colors of CGO components
    '''
    def __init__(self, parent, **kwargs):
        self.parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "ComponentColorButtons: ", arg, type(getattr(self, arg))

        self.color11_label = Tkinter.Label(
            parent,
            text = '11',
        )
        self.color22_label = Tkinter.Label(
            parent,
            text = '22',
        )
        self.color33_label = Tkinter.Label(
            parent,
            text = '33',
        )
        self.color11_label.grid(
            row = 0,
            column = 0,
            sticky = 'ew',
        )
        self.color22_label.grid(
            row = 1,
            column = 0,
            sticky = 'ew',
        )
        self.color33_label.grid(
            row = 2,
            column = 0,
            sticky = 'ew',
        )

        self.color11_button = Tkinter.Button(parent,
            command = lambda: self.get_color(
                self.color11, 
                self.color11_button
            ),
            bg = self.color11.get_hex_color(),
            width = 10,
        )
        self.color22_button = Tkinter.Button(parent,
            command = lambda: self.get_color(
                self.color22, 
                self.color22_button
            ),
            bg = self.color22.get_hex_color(),
        )   
        self.color33_button = Tkinter.Button(parent,
            command = lambda: self.get_color(
                self.color33, 
                self.color33_button
            ),
            bg = self.color33.get_hex_color(),
        )

        self.color11_button.grid(row = 0,
            column = 1,
            sticky = 'ew',
        )
        self.color22_button.grid(row = 1,
            column = 1,
            sticky = 'ew',
        )
        self.color33_button.grid(row = 2,
            column = 1,
            sticky = 'ew',
        )

    def get_color(self, color_class, color_button):
        (rgbTuple, rgbHex) = tkColorChooser.askcolor(
            color_class.get_hex_color()
        )
        if rgbTuple != None:
            color_class.set_color(rgbTuple)
            color_class.normalize()
            color_button.configure(bg = color_class.get_hex_color())


class CGOColor:
    '''
    a wrapper class defining colors of CGOs and passing them around.
    Also facilitates the normalization of color code and conversion to Hex 
    code for use with Tk
    '''
    def __init__(self, color = [0.0, 0.0, 0.0]):
        self.color = list(color)

        self.rgb_to_hex()

    def set_color(self, rgb = [0.0, 0.0, 0.0]):
        self.color = list(rgb)
        self.rgb_to_hex()
        
    def rgb_to_hex(self):
        r8 = int(self.color[0] * 65535) >> 8 
        g8 = int(self.color[1] * 65535) >> 8
        b8 = int(self.color[2] * 65535) >> 8
        # print "#%02X%02X%02X" % (r8, g8, b8)
        self.hex_color = "#%02X%02X%02X" % (r8, g8, b8)

    def get_hex_color(self):
        return deepcopy(self.hex_color)

    def get_color(self):
        return deepcopy(self.color)

    def normalize(self):
        if (self.color[0] > 1) \
            or (self.color[1] > 1) \
            or (self.color[2] > 1):

            self.color[0] = self.color[0] / 255.0
            self.color[1] = self.color[1] / 255.0
            self.color[2] = self.color[2] / 255.0


            self.rgb_to_hex()

##############################################################################    
#
# Command line interface access to the plugin from PyMOL interpreter
##############################################################################    

def drawcst(
    selection = "all", 
    filename = None, 
    cgo_name = "", 
    width = 0.05, 
    rel_width11 = 1.0, 
    rel_width22 = 1.0, 
    rel_width33 = 1.0, 
    rel_len11 = 1.0,
    rel_len22 = 1.0,
    rel_len33 = 1.0,
    color11 = [1.0, 0.0, 0.0], 
    color22 = [0.00 , 1.0 , 0.0], 
    color33 = [0.00 , 0.00 , 1.0],
    pseudo = 1,
    show_pseudo = 1,
):

    '''
    A handy-dandy script for visualising the results of ab initio 
    calculation of chemical shift tensors performed in Gaussian program. It
    reads the Gaussian output containing the chemical shift tensor eigenvalues
    and eigenvectors (to calculate these run the Gaussian job with
    NMR(PrintEigenvectors) keyword), and draws the eigenvectors as CGO arrows
    for the nuclei of your choice. 

    You have to load the XYZ (PDB, etc...) file
    containing the atomic coordinates used in the calculation.

    USAGE

        drawcst selection, 
            filename = None, 
            [cgo_name = "", 
            [width=0.02, 
            [rel_width11/22/33 = 1.0,
            [rel_len11/22/33 = 1.0,
            [color11 = [1.0, 0.0, 0.0],
            [color22 = [0.0, 1.0, 0.0],
            [color33 = [0.0, 0.0, 1.0],
            [pseudo = 1,
            [show_pseudo = 1,
            ]]]]]]]]]

        where:
           "selection"     specifies the atoms for which to draw NMR tensors

           "filename"    is the name of the Gaussian output file containing 
                           NMR tensors
           
           "cgo_name"       name prefix of the CGOs and pseudoatoms.

           "width"         specifies the width of CGO arrows used for drawing
                           and the size of pseudoatom spheres
           
           "rel_width11/22/33"   relative width of individual components,
                           e.g. rel_width11 = 1.5 makes the arrow 
                           corresponding to 11 component 1.5x thicker relative 
                           to other components

           "rel_len11/22/33"     similar to above, only controls the length of 
                           CGO arrows

           "color11/22/33" colors the individual arrows. Color code is passed
                           as a list of RGB values from 0 to 1

           "pseudo"        controls whether pseudoatoms are placed at the 
                           tips of CGO arrows, drawn as a spheres with radius
                           equal to "width". These pseudoatoms have names
                           "11", "22", "33" and the values of corresponding
                           magnetic shielding components are written in the
                           b-factor property. These pseudoatoms are useful
                           for the labeling of the CGO objects and also for
                           measurement (e.g. you can measure an angle between
                           some bond and  magnetic shielding eigenvector).
                           The value of 0 suppressed the creation of
                           pseudoatoms.

           "show_pseudo"    if >0, then the pseudoatoms are shown as spheres at
                           the tips of CGO arrows. 0 makes them hidden

       EXAMPLE
           
           # load an XYZ coordinates of theobromine in standard orientation
           load theobr.xyz, xyz

           # draw the NMR shielding tensor for atom C8, make the CGO representing
           # 11 components 1.5 times thicker and longer, color it black, don't 
           # create pseudoatoms

           drawcst (xyz and name C8), theobr_NMRTensors.log, rel_width11 = 1.5,
               rel_len11 = 1.5, color11 = [0, 0, 0], pseudo=0

    '''

   # sanity check of input values
   # dictionary of value types

    scalar_opt_types = {
        "cgo_name" : "str",
        "width" : "float",
        "pseudo" : "int",
        "rel_width11" : "float",
        "rel_width22" : "float",
        "rel_width33" : "float",
        "rel_len11" : "float",
        "rel_len22" : "float",
        "rel_len33" : "float",
    }

    list_opt_typess = {
        "color11" : "float",
        "color22" : "float",
        "color33" : "float"
    }


    for (opt, opt_type) in scalar_opt_types.iteritems():
        try:
            exec("%s = %s(%s)" % ( opt, opt_type, opt))

        except ValueError:
            print "Option \"%s\" must of type \"%s\"! (Was \"%s\")!"\
            % (opt, opt_type, repr(type(opt)))
            print exc_info()[:1]
            return


    for (opt, opt_type) in list_opt_typess.iteritems():
        #i = eval("type(%s)" % opt)
        #print "before:", i
        try:
            toList = eval("%s" % opt)
            exec("%s = %s" % (opt, toList))

            #i = eval("type(%s)" % opt)
            #print "after: ", i

            for (i, val) in enumerate(eval("%s" % opt)):
                # print i, val
                exec("%s[%s] = %s(%s)" % (opt, i, opt_type, val))
                # print "Type: ", eval("type(%s[%s])" % (opt, i))

        except (NameError, ValueError):
            print "Option \"%s\" must be list of type \"%s\"!" % (opt, opt_type)
            return

    # set the relative widths array for emphasizing certain components
    settings_dict = {
        'cgo_width' : width,
        'cgo_rel_widths' : [rel_width11, rel_width22, rel_width33],
        'cgo_rel_lengths' : [rel_len11, rel_len22, rel_len33],
        'cgo_colors' : [color11, color22, color33]
    }

    if pseudo != 0:
        settings_dict['draw_pseudo'] = True
    else:
        settings_dict['draw_pseudo'] = False

    if show_pseudo != 0:
        settings_dict['show_pseudo'] = True
    else:
        settings_dict['show_pseudo'] = False


    # OK, input is tested, now get to work
    tensors = TensorList()

    try:
        tensors.read(filename)
    except IOError:
        print "Cannot read Gaussian output \"%s\"!" % filename
        print "Please check whether this file exists and is valid!"
        return


    tensors.set_selection(sele = selection, seleName = cgo_name)
    tensors.filter_selection()

    tensors.apply_cgo_settings(settings_dict)

    tensors.redraw_items()

cmd.extend("drawcst", drawcst)

#############################################################################
#
# initialization of the plugin, adding it to the PyMOL > Plugin dir
#############################################################################

def __init__(self):
    """
    Adds CSTViz GUI to the Plugins menu.
    """
    self.menuBar.addmenuitem('Plugin',
        'command',
        'CST vizualization plugin',
        label = 'CSTViz',
        command = lambda s=self: CSTVizGUI(s)
    )
