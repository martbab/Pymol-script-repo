'''
---CSTViz Magnetic shielding tensor visualization plugin for PyMOL---

Author: Martin Babinsky <martbab@chemi.muni.cz>
Date: March 2012

------

TODO: 
    1.) Fix the weird issue with PyMOL main window freezing when switching 
        desktops on Unix system

    2.) Make the plugin load the molecular geometry along the CST data directly 
        from the Gaussian ouput file.

    3.) enable the coloring of pseudoatoms by their corresponding components
        currently they are colored only as red/green/blue

    4.) make the plugin store and retrieve user-defined settings in a config ]
        file

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

    1.) When the plugin is running and user switches desktops or shades window
        the Pymol main window becomes unresponsive and all widget disappear

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
from pymol import cmd, stored
from pymol.cgo import *

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
        eigVec11 = [0.0, 0.0, 0.0], 
        eigVec22 = [0.0, 0.0, 0.0], 
        eigVec33 = [0.0, 0.0, 0.0]
    ):

        self.__sigmas = array([sigma11, sigma22, sigma33])

        self.__eigVecs = array([eigVec11, eigVec22, eigVec33])

        self.__origin = array(origin)

        self.__nucleus = nucleus
        self.__index = index 
        self.__cgoWidth = 0.02
        self.__cgoRelWidths = [ 1.0, 1.0, 1.0 ]
        self.__cgoColors = [[ 1.0, 0.0, 0.0 ],
                            [ 0.0, 1.0, 0.0 ],
                            [ 0.0, 0.0, 1.0 ]]

        self.__drawPseudo = True
        self.__showPseudo = True
        self.__cgoObject = []
        self.__objName = self.__nucleus
        self.__pseudoName = self.__objName + "_comp"

    def setNucleus(self, nuc = ""):
        '''set the information about nucleus type for which the tensor 
        was calculated. Accepts the string with the element symbol'''
        self.__nucleus = nuc

    def setIndex(self, i = 0):
        '''
        sets the index of tensor, for lookup in molecular geometry
        '''
        self.__index = 0

    def getIndex(self):
        '''
        returns the current index of tensor
        '''
        return self.__index

    def getNucleus(self):
        '''returns the string containing the nucleus type'''
        return self.__nucleus


    def getOrigin(self):
        '''
        gets the position of tensor
        '''
        return self.__origin

    def setSigmas(self, vals = [0.0, 0.0, 0.0]):
        '''
        Sets the values of all three sigma eigenvalues at once
        '''
        self.__sigmas = array(vals)

    def getSigmas(self):
        '''
        Returns the array with all three sigma eigenvalues
        '''
        return self.__sigmas

    def setSigma11(self, val = 0.0):
        '''Sets the value of sigma_11'''
        self.__sigmas[0] = val

    def setSigma22(self, val = 0.0):
        '''Sets the value of sigma_22'''
        self.__sigmas[1] = val

    def setSigma33(self, val = 0.0):
        '''Sets the value of sigma_33'''
        self.__sigmas[2] = val

    def getSigma11(self):
        '''Returns the value of sigma_11'''
        return self.__sigmas[0] 

    def getSigma22(self):
        '''Returns the value of sigma_22'''
        return self.__sigmas[1]

    def getSigma33(self):
        '''Returns the value of sigma_33'''
        return self.__sigmas[2]


    def setEigVecs(self, vecs = [[0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0]]):
        '''
        Sets the coordinates of all three sigma eigenvectors at once
        '''
        self.__eigVecs = array(vecs)

    def getEigVecs(self):
        '''
        Returns a 3x3 list of sigma eigenvectors
        '''
        return self.__eigVecs

    def setEigVec11(self, vec = [0.0, 0.0, 0.0]):
        '''sets the values of eigenvector 11'''
        self.__eigVecs[0] = array(vec)

    def setEigVec22(self, vec = [0.0, 0.0, 0.0]):
        '''sets the values of eigenvector 22'''
        self.__eigVecs[1] = array(vec)

    def setEigVec33(self, vec = [0.0, 0.0, 0.0]):
        '''sets the values of eigenvector 33'''
        self.__eigVecs[2] = array(vec)

    def getEigVec11(self):
        '''returns the eigenvector 11 as Numpy array'''
        return self.__eigVecs[0] 

    def getEigVec22(self):
        '''returns the eigenvector 22 as Numpy array'''
        return self.__eigVecs[1]

    def getEigVec33(self):
        '''returns the eigenvector 33 as Numpy array'''
        return self.__eigVecs[2]

    def getSigmaIso(self):
        '''Returns the value of isotropic chemical shielding'''
        return (1 / 3.0) * sum(self.__sigmas)

    def scaleEigVecs(self, values = [1.0, 1.0, 1.0]):
        '''Scales the eigenvectors by arbitrary values'''
        self.__eigVecs[0] *= values[0]
        self.__eigVecs[1] *= values[1]
        self.__eigVecs[2] *= values[2]

    def translateTensor(self, point = [0.0, 0.0, 0.0]):
        '''
        translates the eigenvectors to a given point in space 
        (for example to the coordinates of the corresponding atom)
        '''

        trVector = array(point) - self.__origin
        self.__eigVecs += trVector
        self.__origin = array(point)

    def printInfo(self):
        '''
        prints out a short summary about the tensor parameters
        '''
        message = "Nucleus: "
        message += self.__nucleus
        message += "\n"

        message += "Eigenvalues: " + repr(self.__sigmas) + "\n"
        message += "Isotropic shielding " + repr(self.getSigmaIso()) + "\n"
        message += "Eigenvectors: " + repr(self.__eigVecs) + "\n"
        message += "CGO object width: " + repr(self.__cgoWidth) + "\n"
        message += "Relative widths: " + repr(self.__cgoRelWidths) + "\n"
        message += "Colors: " + repr(self.__cgoColors) + "\n"
        message += "Draw pseudoatoms: " + repr(self.__drawPseudo) + "\n"
        message += "Show pseudoatoms as spheres: " + repr(self.__showPseudo)

        print message

    def setCGOParams(self, cgoWidth=0.02, 
        cgoRelWidths = [ 1.0, 1.0, 1.0 ], 
        cgoColors = [[1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ]],
        drawPseudo = True,
        showPseudo = True
    ):
        '''
        Sets some parameters for the 3-D rendering of shielding tensor
        as a collection of CGOs. Sets up the colors of individual 
        components, overall width of CGOs and relative widths of 
        individual components. Also controls the presentation and drawing 
        of pseudoatoms at the tips of CGO arrows
        '''

        self.__cgoWidth = cgoWidth
        self.__cgoRelWidths = cgoRelWidths
        self.__cgoColors = cgoColors
        self.__drawPseudo = drawPseudo
        self.__showPseudo = showPseudo

    def setCGOWidth(self, width = 0.02):
        '''
        Individual method to set the width of CGO
        '''
        self.__cgoWidth = width
        
    def setCGORelWidth11(self, width = 1.0):
        '''Sets the relative width of CGO representation of component 11'''
        self.__cgoRelWidths[0] = width

    def setCGORelWidth22(self, width = 1.0):
        '''Sets the relative width of CGO representation of component 22'''
        self.__cgoRelWidths[1] = width

    def setCGORelWidth33(self, width = 1.0):
        '''Sets the relative width of CGO representation of component 33'''
        self.__cgoRelWidths[2] = width

    def setCGOColor11(self, color = [ 0.0, 0.0, 0.0 ]):
        '''Sets the color of CGO representation of component 11'''
        self.__cgoColors[0] = color

    def setCGOColor22(self, color = [ 0.0, 0.0, 0.0 ]):
        '''Sets the color of CGO representation of component 22'''
        self.__cgoColors[1] = color

    def setCGOColor33(self, color = [ 0.0, 0.0, 0.0 ]):
        '''Sets the color of CGO representation of component 33'''
        self.__cgoColors[2] = color33

    def setObjName(self, objName):
        '''sets the name of CGO in PyMOL'''
        self.__objName = objName + "_CST"
        self.__pseudoName = self.__objName + "_comp"

    def showPseudo(self, show = True):
        '''controls whether the pseudoatoms at the tips of CGO arrows are 
        visible or hidden'''
        self.__showPseudo = show
   
    def drawPseudo(self, draw = True):
        '''controls the creation of pseudoatoms at the tips of CGO arrows.
        This is very useful for geometrical analysis of tensor components,
        e. g. you can easily measure an angle between specific component and 
        a bond in molecule.

        In addition, the value of the corresponding chemical shielding 
        eigenvalue is written in the b-factor property of pseudoatom'''
        self.__drawPseudo = draw

    def getObjName(self):
        '''returns a name of CGO'''
        return self.__objName

    def getPseudoName(self):
        '''returns the name of pseudoatoms'''
        return self.__pseudoName

    def deleteCGOObject(self):
        '''deletes the CGO Object'''
        cmd.delete(self.__objName)
        cmd.delete(self.__pseudoName)

    def prepareCGOObject(self):
        '''Precalculates the 3D coordinates and colors of CGO representation of 
        shielding tensor'''
        self.__cgoObject = []

        for (n, vec) in enumerate(self.__eigVecs):


            # apply any relative width factor
            cgoWidth = self.__cgoRelWidths[n] * self.__cgoWidth

            # defintion of cone length and end point
            dirVec = vec - self.__origin
            dirVecLength = norm(dirVec)
            coneLength = 5 * cgoWidth

            # cylinder endpoint definition, made in a way that seems
            # to produce cones that are invariant under scaling
            cylinderEndPoint = ((dirVec / dirVecLength)\
                * (dirVecLength - coneLength))

            # end of cones at eigenvector position and its reflection
            # around coordinates of nucleus
            cone1End = vec
            cone2End = 2 * self.__origin - vec

            # first define that we draw a cylinder
            self.__cgoObject.append(CYLINDER)

            # the origin of cylinder is at reflected endpoint
            # around nuclear
            self.__cgoObject.extend(self.__origin - cylinderEndPoint)

            # the end of cylinder is at the endpoint
            self.__cgoObject.extend(self.__origin + cylinderEndPoint)

            # the width of cylinder
            self.__cgoObject.append(cgoWidth)

            # the first and second colors are the same
            self.__cgoObject.extend(self.__cgoColors[n] * 2)

            #then draw a cone
            self.__cgoObject.append(CONE)

            #self.__origin of the cone is at the end of cylinder
            self.__cgoObject.extend(self.__origin + cylinderEndPoint)

            #cone ends at the point of the eigenvector
            self.__cgoObject.extend(cone1End)

            # diameter equals to 1.618 * width
            self.__cgoObject.append(1.618 * cgoWidth)

            # no idea what this number means
            self.__cgoObject.append(0.0)

            # colors, hopefully
            self.__cgoObject.extend(self.__cgoColors[n] * 2)

            # ummm, no idea
            self.__cgoObject.extend([1.0, 1.0])


            #then draw another cone
            self.__cgoObject.append(CONE)

            #origin of the cone is at the origin of cylinder
            self.__cgoObject.extend(self.__origin - cylinderEndPoint)

            #cone ends at the reflected position of the eigenvector
            self.__cgoObject.extend(cone2End)

            # diameter equals to 1.618 * width
            self.__cgoObject.append(1.618 * cgoWidth)

            # no idea what this number means
            self.__cgoObject.append(0.0)

            # colors, hopefully
            self.__cgoObject.extend(self.__cgoColors[n] * 2)

            # ummm, no idea
            self.__cgoObject.extend([1.0, 1.0])


    def drawCGOObject(self):
        '''Do the actual drawing in the PyMol scene and create pseudoatoms if 
        requested'''
        # delete previously drawn CGO
        cmd.delete(self.__objName)
        cmd.load_cgo(self.__cgoObject, self.__objName)
        cmd.delete(self.__pseudoName)

        if self.__drawPseudo:
            self.__pseudoRadius = 1.5 * self.__cgoWidth
            cmd.pseudoatom(self.__pseudoName, name="11", 
                vdw = self.__pseudoRadius,
                b = self.__sigmas[0],
                pos = tuple(self.__eigVecs[0]), color="red")
            cmd.pseudoatom(self.__pseudoName, name="22", 
                vdw = self.__pseudoRadius,
                b = self.__sigmas[1],
                pos = tuple(self.__eigVecs[1]), color="green")
            cmd.pseudoatom(self.__pseudoName, name="33", 
                vdw = self.__pseudoRadius,
                b = self.__sigmas[2],
                pos = tuple(self.__eigVecs[2]), color="blue")
            cmd.show_as("spheres", self.__pseudoName)

            if not self.__showPseudo:
                cmd.hide("spheres", self.__pseudoName)



class TensorList(dict):
    '''
    Container of SigmaTensor objects, facilitates reading tensor information from
    Gaussian output class
    '''

    def __init__(self):
        self = {}


    def __parseGauOutFile(self, gauOutfile):
        '''Internal method for parsing the Gaussian logfile and storing the 
        results of NMR calculations as SigmaTensor objects'''
        # TODO: this stuff is a mess, try to rewrite it in a cleaner way

        # this string signals the start of NMR properties section in logfile
        sectionStart = "Magnetic shielding tensor (ppm)"
        # this substring denotes the end
        sectionEnd = "End of Minotr Frequency-dependent properties"
        atomKey = ""
        sectionFound = False
        eigVectorsFound = False
        vectors = []
        lineNum = 0

        for line in gauOutfile:
            lineNum += 1
            # print lineNum 

            if sectionEnd in line:
                break

            if sectionStart in line:
                sectionFound = True
                continue

            if sectionFound:
                fields = line.split()
                if eigVectorsFound:                
                    # print fields
                    vectorIndex = int(fields[0][1])
                    if vectorIndex != 3:
                        vectors.append([float(n) for n in fields[1:4]])
                    else:
                        vectors.append([float(n) for n in fields[1:4]])
                        self[atomKey].setEigVecs(vectors)
                        eigVectorsFound = False
                        continue

                # print fields
                if "Isotropic" in fields:
                    atomKey = "%s%03d" % (fields[1], int(fields[0]))
                    # print atomKey
                    self[atomKey] = SigmaTensor(nucleus = fields[1], 
                        index = int(fields[0]))
                    self[atomKey].setObjName(atomKey)
                elif "Eigenvalues:" in fields:
                    self[atomKey].setSigmas([float(fields[1]), float(fields[2]),
                        float(fields[3])])
                elif "Eigenvectors:" in fields:
                    eigVectorsFound = True
                    vectors = []
                    continue
           



            
    def read(self, gauOutputName = ""):
        '''Method to read the Gaussian log file and store the results of NMR 
        calculation as a dictionary of SigmaTensor objects'''
        gauOut = open(gauOutputName, "r")

        self.__parseGauOutFile(gauOut)

        gauOut.close()
        

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

class CSTVizGUI:
    def __init__(self, app):
        self.__fileList = {}
        self.__moleculeList = {}
        self.__parent = app.root
        self.__redrawText = 'Redraw/refresh'
        self.__exitText =  'Exit'

        self.__drawPseudoVar = Tkinter.BooleanVar()
        self.__showPseudoVar = Tkinter.BooleanVar()
        self.__absCGOWidthVar = Tkinter.DoubleVar()

        self.__relWidth11Var = Tkinter.DoubleVar()
        self.__relWidth22Var = Tkinter.DoubleVar()
        self.__relWidth33Var = Tkinter.DoubleVar()

        self.__relLen11Var = Tkinter.DoubleVar()
        self.__relLen22Var = Tkinter.DoubleVar()
        self.__relLen33Var = Tkinter.DoubleVar()

        self.__color11 = CGOColor([1, 0, 0])
        self.__color22 = CGOColor([0, 1, 0])
        self.__color33 = CGOColor([0, 0, 1])

        self.__settingsDictionary = {
            'drawPseudoVar' : self.__drawPseudoVar,
            'showPseudoVar' : self.__showPseudoVar,
            'absCGOWidthVar' : self.__absCGOWidthVar,
            'relWidth11Var' : self.__relWidth11Var,
            'relWidth22Var' : self.__relWidth22Var,
            'relWidth33Var' : self.__relWidth33Var,
            'relLen11Var' : self.__relLen11Var,
            'relLen22Var' : self.__relLen22Var,
            'relLen33Var' : self.__relLen33Var,
            'color11' : self.__color11,
            'color22' : self.__color22,
            'color33' : self.__color33
        }

        self.__selectedFile = ""
        self.__selectedFileIndex = 0

        self.__selectedCSTs = []

        #set the default values of variables
        self.__setDefaults()


        #####################################################################
        # The main window configuration
        # with label on top and bottom buttons
        #
        #####################################################################
        self.__appTitle = 'NMR Shielding Tensors visualization plugin'
        self.__topLabelText = "(c) 2012 Martin Babinsky: \
                martbab (at) chemi (dot) muni (dot) cz"
        self.__top = Pmw.Dialog(self.__parent,
            buttons = (self.__redrawText, self.__exitText),
            command = self.execute
        )
        self.__top.title(self.__appTitle)
        self.__top.withdraw()

        self.__top.component('hull').geometry('600x400')
        Pmw.setbusycursorattributes(self.__top.component('hull'))
        self.__top.protocol("WM_DELETE_WINDOW", self.exit)
        self.__topLabel = Tkinter.Label(self.__top.interior(), 
            text = self.__topLabelText
        )
        self.__topLabel.grid(row = 0, 
            column = 0, 
            sticky="we"
        )
#       self.__redrawButton = Tkinter.Button(self.__top.interior(),
#           text = self.__redrawText,
#           command = self.redrawCSTs
#       )
#       self.__redrawButton.grid(
#           row = 2,
#           column = 0,
#       )
#       self.__exitButton = Tkinter.Button(self.__top.interior(), 
#           text = self.__exitText,
#           command = self.exit
#       )
#       self.__exitButton.grid(
#           row = 2,
#           column = 1,
#       )
        #####################################################################
        # the notebook class to organize GUI into two groups
        # 1) List of loaded files/tensors, adding/deleting items, configuring
        #    individual items
        # 2) Global settings
        #####################################################################
        self.__noteBook = Pmw.NoteBook(self.__top.interior())
        self.__noteBook.grid(
            column = 0,
            row = 1,
            sticky = "nsew"
        )

        self.__top.interior().grid_columnconfigure(0,
            weight = 1
        )
        self.__top.interior().grid_rowconfigure(1,
            weight = 1,
        )
        self.__cstManager = self.__noteBook.add('CST list')
        self.__globalSettings = self.__noteBook.add('Global settings')

        #####################################################################
        # CST Manager controls loading/deleting of new items and individual
        # configuration
        #
        #####################################################################
        self.__cstManager.rowconfigure(0,
            weight = 1
        )
        self.__cstManager.columnconfigure(0,
            weight = 1
        )

        self.__cstManagerHandler(self.__cstManager)

        #####################################################################
        # The general settings section controls the default global settings
        # these can be overriden by individual settings
        #####################################################################
        self.__globalSettingsHandler = SettingsWindow(self.__globalSettings,
            applyCommand = self.applyGlobalSettings,
            setDefaultCommand = self.__setDefaults,
            **self.__settingsDictionary
#           drawPseudoVar = self.__drawPseudoVar,
#           showPseudoVar = self.__showPseudoVar,
#           absCGOWidthVar = self.__absCGOWidthVar,

#           relWidth11Var = self.__relWidth11Var,
#           relWidth22Var = self.__relWidth22Var,
#           relWidth33Var = self.__relWidth33Var,

#           relLen11Var = self.__relLen11Var,
#           relLen22Var = self.__relLen22Var,
#           relLen33Var = self.__relLen33Var,

#           color11 = self.__color11,
#           color22 = self.__color22,
#           color33 = self.__color33,
        )

        self.showAppModal()

    def __cstManagerHandler(self, parent):
        self.__filesItemsWindow(parent)

        #self.__cstButtonFrame = Tkinter.Frame(parent)
        #self.__cstButtonFrame.grid(
        #    row = 0,
        #    column = 1,
        #)

        #self.__cstButtonGroup(self.__cstButtonFrame)


    def __filesItemsWindow(self, parent):
        self.__fileListFrame = Tkinter.LabelFrame(parent,
            text = "List of loaded files"
        )
        self.__fileListFrame.grid(row = 0,
            column = 0,
            sticky = "nsew"
        )
        self.__fileListFrame.columnconfigure(0, 
            weight = 1,
        )
        self.__fileListFrame.rowconfigure(0, 
            weight = 1,
        )

        self.__fileContentsFrame = Tkinter.LabelFrame(parent,
            text = "file contents"
        )

        self.__fileContentsFrame.grid(row = 0,
            column = 1,
            sticky = "nsew"
        )
        self.__fileContentsFrame.columnconfigure(0, 
            weight = 1,
        )
        self.__fileContentsFrame.rowconfigure(0, 
            weight = 1,
        )
        self.__fileListComponent(self.__fileListFrame)
        self.__fileContentsListBoxComponent(self.__fileContentsFrame)
        



    def __fileListComponent(self, parent):
        self.__fileListBox = Pmw.ScrolledListBox(parent,
#           label_text = "Loaded files",
#           labelpos = 'nw',
            listbox_height = 8,
            selectioncommand = self.selectFile,
#            listbox_selectmode = "extended",
        )

        self.__fileListBox.grid(
            row = 0,
            column = 0,
            columnspan = 1,
#           rowspan = 3,
            sticky = 'nsew'
        )

        self.__fileListButtonFrame = Tkinter.Frame(parent)
        self.__fileListButtonFrame.grid(
            row = 0,
            column = 1,
            sticky = 'nsew'
        )

        self.__fileListButtonGroup(self.__fileListButtonFrame)
        self.__selectionFilterFrame = Tkinter.LabelFrame(parent,
            text = "data filtering",
        )
        self.__selectionFilterFrame.grid(
            row = 4,
            column = 0, 
            columnspan = 2,
            sticky = 'nsew'
        )

        self.__selectionFilterGroup(self.__selectionFilterFrame)
        
        ###########################
        # menu for file list
        ###########################

        self.__fileListBoxMenu = Tkinter.Menu( parent,
            tearoff = 0,
        )

        self.__fileListBoxMenu.add_command( label = "Add file(s)",
            command = self.openFile,
        )
        self.__fileListBoxMenu.add_command( label = "Reload File",
            command = self.reloadFile,
        )
        self.__fileListBoxMenu.add_command( label = "Assign molecule object",
            command = self.assignMolToCSTs,
        )
        self.__fileListBoxMenu.add_command( label = "Limit to selection",
            command = self.filterToSelection
        )
        self.__fileListBoxMenu.add_command( label = "Settings",
            command = lambda: self.localSettingsWindow(parent),
        )
        self.__fileListBoxMenu.add_command( label = "Remove file",
            command = self.removeFile,
        )
        ###########################
        self.__fileListBox.component("listbox").bind("<Button-3>", self.fileMenu)

    def __fileListButtonGroup(self, parent):
        self.__loadFileButton = Tkinter.Button(parent,
            text = "Open file(s)",
            command = self.openFile,
        )
        self.__loadFileButton.grid(
            row = 0,
            column = 0,
            sticky='we'
        )

        self.__reloadFileButton = Tkinter.Button(parent,
            text = "Reload file",
            command = lambda: self.localSettingsWindow(parent),
        )
        self.__reloadFileButton.grid(
            row = 2,
            column = 0,
            sticky = 'we'
        )
        self.__removeFileButton = Tkinter.Button(parent,
            text = "Remove file",
            command = self.removeFile,
        )

        self.__removeFileButton.grid(
            row = 3,
            column = 0,
            sticky = 'we'
        )

        self.__removeAllFilesButton = Tkinter.Button(parent,
            text = "Remove all files",
        )

        self.__removeAllFilesButton.grid(
            row = 4,
            column = 0,
            sticky = 'we'
        )

    def __selectionFilterGroup(self, parent):
        self.__selectionEntry = Pmw.EntryField( parent,
            command = self.filterToSelection,
            label_text = "selection",
            labelpos = 'nw',
            value = 'all',
        )
        self.__selectionEntry.grid(
            row = 0,
            column = 0,
            sticky = 'we'
        )
        self.__selectionNameEntry = Pmw.EntryField( parent,
            command = self.filterToSelection,
            label_text = "name for selection",
            labelpos = 'nw',
            value = 'sele1',
        )
        self.__selectionNameEntry.grid(
            row = 1,
            column = 0,
            sticky = 'we'
        )

        self.__getSelectionButton = Tkinter.Button( parent,
            text = 'get PyMOL object/selection',
            command = self.getPyMolObjSele,
        )
        self.__getSelectionButton.grid(
            row = 2,
            column = 0,
            sticky = 'we'
        )
        self.__filterSelectionButton = Tkinter.Button( parent,
            text = 'Filter data',
            command = self.filterToSelection,
        )
        self.__filterSelectionButton.grid(
            row = 3,
            column = 0,
            sticky = 'we'
        )
    def __fileContentsListBoxComponent(self, parent):
        self.__fileContentsListBox = Pmw.ScrolledListBox(self.__fileContentsFrame,
#           label_text = "CSTs",
#           labelpos = 'nw',
            listbox_height = 10,
            listbox_selectmode = "extended",
        )
        self.__fileContentsListBox.grid(
            row = 0,
            column = 0,
            sticky = 'nsew'
        )
        self.__fileContentsButtonsFrame = Tkinter.Frame(parent)
        self.__fileContentsButtonsFrame.grid(
            row = 0,
            column = 1,
            sticky = 'nsew'
        )
        self.__fileContentsButtonsGroup(self.__fileContentsButtonsFrame)

        ###########################
        # menu for CSTs
        ###########################
        self.__dataFileContentsMenu = Tkinter.Menu(parent, tearoff = 0)
        self.__dataFileContentsMenu.add_command(label = "Settings", 
            command = lambda: self.localSettingsWindow(parent),
        )
        self.__dataFileContentsMenu.add_command(label = "Remove",
            command = self.removeSel,
        )
        self.__dataFileContentsMenu.add_command(label = "Clear List",
            command = self.clearAllCSTs,
        )
        ###########################
        self.__fileContentsListBox.component("listbox").bind("<Button-3>", 
            self.cstMenu
        )

    def __fileContentsButtonsGroup(self, parent):
        self.__graphicalSettingsButton = Tkinter.Button( parent,
            text = "Edit",
            command = lambda: self.localSettingsWindow(parent),
        )

        self.__graphicalSettingsButton.grid(
            row = 0,
            column = 0,
            sticky = 'ew'
        )
        self.__showInfoButton = Tkinter.Button( parent,
            text = "Show info",
        )
        self.__showInfoButton.grid(
            row = 1,
            column = 0,
            sticky = 'ew',
        )
        self.__reloadListButton = Tkinter.Button( parent,
            text = "Reload list",
            command = self.reloadFile,
        )
        self.__reloadListButton.grid(
            row = 2,
            column = 0,
            sticky = 'ew',
        )
        self.__removeButton = Tkinter.Button( parent,
            text = "Remove",
            command = self.removeSel,
        )
        self.__removeButton.grid(
            row = 3,
            column = 0,
            sticky = 'ew'
        )
        self.__removeAllButton = Tkinter.Button( parent,
            text = "Remove all",
            command = self.clearAllCSTs,
        )
        self.__removeAllButton.grid(
            row = 4,
            column = 0,
            sticky = 'ew',
        )


    def cstMenu(self, event):
        self.__dataFileContentsMenu.tk_popup(event.x_root, event.y_root)
        self.__selectedCSTs = self.__fileContentsListBox.getcurselection()

    def fileMenu(self, event):
        self.__fileListBoxMenu.tk_popup(event.x_root, event.y_root)

        try:
            self.__selectedCSTs = self.__fileList[self.__selectedFile].keys()
        except KeyError:
            pass

    def selectFile(self):
        try:
            self.__selectedFile = self.__fileListBox.getcurselection()[0]
            self.updateCSTView()
            self.__selectedCSTs = self.__fileList[self.__selectedFile].keys()
        except IndexError, KeyError:
            pass

    def openFile(self):
        fileList = tkFileDialog.askopenfilenames(
            parent = self.__fileListFrame,
            title = "Choose a Gaussian output"
        )
        for f in fileList:
            if not f in self.__fileList:
                self.__fileList[f] = TensorList()
                #print basename(f)
                self.__fileList[f].read(f)
                self.__fileListBox.insert("end", f)

                for tensor in self.__fileList[f]:
                    self.applyCGOSettings(self.__fileList[f][tensor])

                #print repr(self.__fileList[basename(f)])

    def reloadFile(self):
        try:
            self.clearAllCSTs()
            self.__fileList[self.__selectedFile].read(self.__selectedFile)

            for tensor in self.__fileList[self.__selectedFile]:
                self.applyCGOSettings(
                    self.__fileList[self.__selectedFile][tensor]
                )

            self.updateCSTView()
        except KeyError, IndexError:
            pass

    def assignMolToCSTs(self):
        molecules = cmd.get_names_of_type('object:molecule')

        self.__molAssiComboBox = Pmw.ComboBoxDialog(parent,
            title = 'molecule assignment',
            label_text = 'assign molecular geometry to data',
            combobox_labelpos = 'n',
            buttons = ('Assign', 'Cancel'),
            scrolledlist_items = tuple(molecules),
        )
        self.__molAssiComboBox.withdraw()
        result = self.__molAssiComboBox.activate()
        molecule = self.__molAssiComboBox.get()

        if molecule != "":
            # print molecule
            atomList = cmd.get_model(molecule).atom
            atomDict = {}

            for atom in atomList:
                atomDict[atom.index] = atom


            for t in self.__fileList[self.__selectedFile]:
                self.__fileList[self.__selectedFile][t].deleteCGOObject()

                try:
                    self.__fileList[self.__selectedFile][t].setObjName(molecule + "_" + t)

                    self.__fileList[self.__selectedFile][t].translateTensor(
                        atomDict[self.__fileList[self.__selectedFile][t].getIndex()].coord,
                    )
                    self.__fileList[self.__selectedFile][t].prepareCGOObject()
                    # print self.__fileList[self.__selectedFile][t].getOrigin()

                except IndexError, KeyError:
                    print "No chemical shielding information for atom %s" % t

        self.updateCSTView()

    def getPyMolObjSele(self):
        selections = cmd.get_names('all')

        self.__filterToSeleComboBox = Pmw.ComboBoxDialog(self.__top,
            title = 'filter data by molecule/selection',
            label_text = 'choose PyMOL object',
            combobox_labelpos = 'n',
            buttons = ('Limit', 'Cancel'),
            scrolledlist_items = tuple(selections),
        )
        self.__filterToSeleComboBox.withdraw()
        result = self.__filterToSeleComboBox.activate()
        selection = self.__filterToSeleComboBox.get()

        if selection != "":
            self.__selectionEntry.setvalue(selection)

    def filterToSelection(self):

        selection = self.__selectionEntry.getvalue()
        selectionName = self.__selectionNameEntry.getvalue()
        atomList = cmd.get_model(selection).atom

        atomDict = {}
        
        dataToDelete = []
        for atom in atomList:
            atomDict[atom.index] = atom

        # loop over data in selected file and test which one of them is 
        # contained in selection. If it is not, add it to the group of data
        # to delete
        for t in self.__fileList[self.__selectedFile]:
            if self.__fileList[self.__selectedFile][t].getIndex() in atomDict:
                self.__fileList[self.__selectedFile][t].setObjName(selectionName + "_" + t)
                self.__fileList[self.__selectedFile][t].translateTensor(
                    atomDict[self.__fileList[self.__selectedFile][t].getIndex()].coord
                )
                self.__fileList[self.__selectedFile][t].prepareCGOObject()
            else:
                self.__fileList[self.__selectedFile][t].deleteCGOObject()
                dataToDelete.append(t)

        for d in dataToDelete:
            del self.__fileList[self.__selectedFile][d]

        self.updateCSTView()

    def removeFile(self):
        self.clearAllCSTs()
        del self.__fileList[self.__selectedFile]
        self.__fileListBox.delete("active")

    def updateCSTView(self):
        # print selectedFile
        # print repr(selectedTensor)

        try:
            self.__fileContentsListBox.setlist(
                sorted(self.__fileList[self.__selectedFile].keys())
            )
        except KeyError,IndexError:
            self.__fileContentsListBox.clear()

    def localSettingsWindow(self, parent):
        self.__windowParent = Tkinter.Toplevel(parent)
        self.__localSettings = SettingsWindow(self.__windowParent,
            applyCommand = lambda: self.applyLocalSettings(self.__windowParent),
            setDefaultCommand = self.__setDefaults,
            **self.__settingsDictionary
        )


    def removeSel(self):
        selection = self.__fileContentsListBox.getcurselection()
        for sel in selection:
            self.__fileList[self.__selectedFile][sel].deleteCGOObject()
            del self.__fileList[self.__selectedFile][sel]

        self.updateCSTView()



    def __setDefaults(self):
        self.__drawPseudoVar.set(True)
        self.__showPseudoVar.set(True)
        self.__absCGOWidthVar.set(0.05)

        self.__relWidth11Var.set(1.0)
        self.__relWidth22Var.set(1.0)
        self.__relWidth33Var.set(1.0)

        self.__relLen11Var.set(1.0)
        self.__relLen22Var.set(1.0)
        self.__relLen33Var.set(1.0)

        self.__color11.setColor([1.0, 0.0, 0.0])
        self.__color22.setColor([0.0, 1.0, 0.0])
        self.__color33.setColor([0.0, 0.0, 1.0])
        

    def applyCGOSettings(self, tensor):
        tensor.setCGOParams(
            cgoWidth = self.__absCGOWidthVar.get(),
            cgoRelWidths = [ 
                self.__relWidth11Var.get(),
                self.__relWidth22Var.get(),
                self.__relWidth33Var.get(),
            ],
           cgoColors = [
                self.__color11.getColor(),
                self.__color22.getColor(),
                self.__color33.getColor()
            ],
            drawPseudo = self.__drawPseudoVar.get(),
            showPseudo = self.__showPseudoVar.get()
        )
        tensor.prepareCGOObject()



    def applyLocalSettings(self, parent):
        for sel in self.__selectedCSTs:
            try:
                self.applyCGOSettings(self.__fileList[self.__selectedFile][sel])
            except KeyError:
                print "Atom", sel, "is not present in list!"
                pass
#           print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#           print sel
#           self.__fileList[self.__selectedFile][sel].printInfo()
#           print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        parent.destroy()

    def applyGlobalSettings(self):
        for fileName in self.__fileList:
            for tensor in self.__fileList[fileName]:
                self.applyCGOSettings(self.__fileList[fileName][tensor])                
#               self.__fileList[fileName][tensor].printInfo()

    def clearAllCSTs(self):
        self.__fileContentsListBox.clear()

        if self.__selectedFile in self.__fileList:
            for tensor in self.__fileList[self.__selectedFile]:
               self.__fileList[self.__selectedFile][tensor].deleteCGOObject()

            self.__fileList[self.__selectedFile].clear()

    def redrawCSTs(self):
        for filename in self.__fileList:
            for i in self.__fileList[filename]:
                self.__fileList[filename][i].drawCGOObject()

    def exit(self, event = None):
        self.__top.destroy()


    def execute(self, event = None):
        if event == self.__redrawText:
            self.redrawCSTs()
        else:
            self.exit()

    def showAppModal(self):
        self.__top.show()
#############################################################################
#
# some parts of GUI written as separate classes to improve readability a bit
#############################################################################
class SettingsWindow:
    '''
    widget for the settings, etiher local or global
    '''
    def __init__(self, parent, 
        **kwargs
    ):
        self.__parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "SettingsWindow", arg, type(getattr(self, arg))

        self.__parent.rowconfigure(0,
            weight = 1
        )
        self.__parent.columnconfigure(0,
            weight = 1
        )
        self.__fileIOSettings = Tkinter.LabelFrame(self.__parent,
            text = "File handling",
        )
        self.__fileIOSettings.grid(row = 0,
            column = 0,
            sticky = 'new',
            columnspan = 2,
        )
        self.__fileIOSettings.columnconfigure(0,
            weight = 1,
        )
        self.__graphicSettings = Tkinter.LabelFrame(self.__parent,
            text = "Graphical settings",
        )
        self.__graphicSettings.grid(row = 1,
            column = 0,
            sticky = 'new',
            columnspan = 2,
        )
        self.__graphicSettings.columnconfigure(1,
            weight = 1,
        )
        self.__fileIOSettingsHandler(self.__fileIOSettings)
        self.__graphSettingsHandler(self.__graphicSettings)
        self.__applyButton = Tkinter.Button(self.__parent,
            text = 'Apply settings',
            command = self.applyCommand,
        )
        self.__applyButton.grid(row = 2,
            column = 0,
            sticky = 'we'
        )
        self.__globalDefaultButton = Tkinter.Button(self.__parent,
            text = 'Restore defaults',
            command = self.setDefaultCommand,
        )
        self.__globalDefaultButton.grid(row = 2,
            column = 1,
            sticky = 'we'
        )

    def __fileIOSettingsHandler(self, parent):
        self.__drawPseudoCheckButton = Tkinter.Checkbutton(parent,
           text = 'Draw pseudoatoms along CGO arrows',
           variable = self.drawPseudoVar,
        )
        self.__drawPseudoCheckButton.grid(
            row = 0,
            column = 0,
            sticky = 'w'
        )

        self.__showPseudoAtomsCheckButton = Tkinter.Checkbutton(parent,
            text = 'Show pseudoatoms',
            variable = self.showPseudoVar,
        )
        self.__showPseudoAtomsCheckButton.grid(
            row = 1,
            column = 0,
            sticky = 'w'
        )

    def __graphSettingsHandler(self, parent):
        self.__absoluteWidthSpinBox = Tkinter.Spinbox(parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.01,
            width = 5,
            textvariable = self.absCGOWidthVar,
        )

        self.__absoluteWidthSpinBox.grid(row = 0,
            column = 0,
            sticky = 'w',
        )
        
        self.__absoluteWidthLabel = Tkinter.Label(parent,
            text = 'Absolute width of CGO arrows',
        )

        self.__absoluteWidthLabel.grid(row = 0,
            column = 1,
            sticky = 'w',
        )
        self.__relWidthGroup = Tkinter.LabelFrame(parent,
            text = 'Relative widths of tensor components',
            borderwidth = 0,
        )
        self.__relWidthGroup.grid(row = 1,
            column = 0, 
            sticky = 'we',
            columnspan = 2
        )
        self.__relWidthsSpinboxGroup = RelWidthsSpinBoxes(self.__relWidthGroup,
            relWidth11Var = self.relWidth11Var,
            relWidth22Var = self.relWidth22Var,
            relWidth33Var = self.relWidth33Var
        )
        #self.__relWidthsScaleGroup(self.__relWidthGroup)
        self.__relLenGroup = Tkinter.LabelFrame(parent,
            text = 'Relative lengths of tensor components',
            borderwidth = 0,
        )
        self.__relLenGroup.grid(row = 2,
            column = 0,
            sticky = 'we',
            columnspan = 2,
        )

        self.__relLenSpinboxGroup = RelLenSpinBoxes(self.__relLenGroup,
            relLen11Var = self.relLen11Var,
            relLen22Var = self.relLen22Var,
            relLen33Var = self.relLen33Var
        )

        self.__colorGroup = Tkinter.LabelFrame(parent,
            text = "Color of tensor components",
            borderwidth = 0,
        )
        self.__colorGroup.grid(row = 1,
            column = 2,
            sticky = 'nw'
        )
        self.__componentColorGroup = ComponentColorButtons(self.__colorGroup,
                color11 = self.color11,
                color22 = self.color22,
                color33 = self.color33
        )

class RelWidthsSpinBoxes:
    '''
    Widget controlling the relative width of CGO components
    '''
    def __init__(self, parent, **kwargs):
        self.__parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "RelWidthsSpinBoxes", arg, type(getattr(self, arg))

        self.__relWidth11SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relWidth11Var,
        )
        self.__relWidth11SB.grid(row = 0,
            column = 0,
            sticky = 'w'
        )

        self.__relWidth22SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relWidth22Var,
        )
        self.__relWidth22SB.grid(row = 1,
            column = 0,
            sticky = 'w'
        )

        self.__relWidth33SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relWidth33Var,
        )
        self.__relWidth33SB.grid(row = 2,
            column = 0,
            sticky = 'w'
        )

        self.__relWidth11Label = Tkinter.Label(self.__parent,
            text = '11',
        )

        self.__relWidth11Label.grid(row = 0,
            column = 1,
            sticky = 'w'
        )

        self.__relWidth22Label = Tkinter.Label(self.__parent,
            text = '22',
        )

        self.__relWidth22Label.grid(row = 1,
            column = 1,
            sticky = 'w'
        )

        self.__relWidth33Label = Tkinter.Label(self.__parent,
            text = '33',
        )

        self.__relWidth33Label.grid(row = 2,
            column = 1,
            sticky = 'w'
        )

class RelLenSpinBoxes:
    '''
    widget controlling the relative length of CGO components
    '''
    def __init__(self, parent, **kwargs):

        self.__parent = parent
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "RelLenSpinBoxes: ", arg, type(getattr(self, arg))

        self.__relLength11SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relLen11Var,
        )
        self.__relLength11SB.grid(row = 0,
            column = 0,
            sticky = 'w'
        )


        self.__relLength22SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relLen22Var,
        )
        self.__relLength22SB.grid(row = 1,
            column = 0,
            sticky = 'w'
        )

        self.__relLength33SB = Tkinter.Spinbox(self.__parent,
            from_ = 0.0,
            to = 10.0,
            increment = 0.05,
            width = 5,
            textvariable = self.relLen33Var,
        )
        self.__relLength33SB.grid(row = 2,
            column = 0,
            sticky = 'w'
        )

        self.__relLength11Label = Tkinter.Label(self.__parent,
            text = '11',
        )

        self.__relLength11Label.grid(row = 0,
            column = 1,
            sticky = 'w'
        )

        self.__relLength22Label = Tkinter.Label(self.__parent,
            text = '22',
        )

        self.__relLength22Label.grid(row = 1,
            column = 1,
            sticky = 'w'
        )

        self.__relLength33Label = Tkinter.Label(self.__parent,
            text = '33',
        )

        self.__relLength33Label.grid(row = 2,
            column = 1,
            sticky = 'w'
        )

class ComponentColorButtons:
    '''
    widget controlling the colors of CGO components
    '''
    def __init__(self, parent, **kwargs):
        self.__parent = parent

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
#           print "ComponentColorButtons: ", arg, type(getattr(self, arg))

        self.__color11Label = Tkinter.Label(parent,
            text = '11',
        )
        self.__color22Label = Tkinter.Label(parent,
            text = '22',
        )
        self.__color33Label = Tkinter.Label(parent,
            text = '33',
        )
        self.__color11Label.grid(row = 0,
            column = 0,
            sticky = 'ew',
        )
        self.__color22Label.grid(row = 1,
            column = 0,
            sticky = 'ew',
        )
        self.__color33Label.grid(row = 2,
            column = 0,
            sticky = 'ew',
        )

        self.__color11Button = Tkinter.Button(parent,
            command = self.__getColor11,
            bg = self.color11.getHexColor(),
            width = 10,
        )
        self.__color22Button = Tkinter.Button(parent,
            command = self.__getColor22,
            bg = self.color22.getHexColor(),
        )   
        self.__color33Button = Tkinter.Button(parent,
            command = self.__getColor33,
            bg = self.color33.getHexColor(),
        )

        self.__color11Button.grid(row = 0,
            column = 1,
            sticky = 'ew',
        )
        self.__color22Button.grid(row = 1,
            column = 1,
            sticky = 'ew',
        )
        self.__color33Button.grid(row = 2,
            column = 1,
            sticky = 'ew',
        )

    def __getColor11(self):
        (rgbTuple, rgbHex) = tkColorChooser.askcolor(
            self.color11.getHexColor()
        )
        if rgbTuple != None:
            self.color11.setColor(rgbTuple)
            self.color11.normalizeColor()
            self.__color11Button.configure(bg = self.color11.getHexColor())

    def __getColor22(self):
        (rgbTuple, rgbHex) = tkColorChooser.askcolor(
            self.color22.getHexColor()
        )
        if rgbTuple != None:
            self.color22.setColor(rgbTuple)
            self.color22.normalizeColor()
            self.__color22Button.configure(bg = self.color22.getHexColor())

    def __getColor33(self):
        (rgbTuple, rgbHex) = tkColorChooser.askcolor(
            self.color33.getHexColor()
        )
        if rgbTuple != None:
            self.color33.setColor(rgbTuple)
            self.color33.normalizeColor()
            self.__color33Button.configure(bg = self.color33.getHexColor())



class CGOColor:
    '''
    a wrapper class defining colors of CGOs and passing them around.
    Also facilitates the normalization of color code and conversion to Hex 
    code for use with Tk
    '''
    def __init__(self, color = [0.0, 0.0, 0.0]):
        self.__color = [0.0, 0.0, 0.0]
        self.__color[0] = color[0]
        self.__color[1] = color[1]
        self.__color[2] = color[2]

        self.__calcHexColor()

    def setColor(self, rgb = [0.0, 0.0, 0.0]):
        self.__color[0] = rgb[0]
        self.__color[1] = rgb[1]
        self.__color[2] = rgb[2]
        self.__calcHexColor()
        
    def __calcHexColor(self):
        r8 = int(self.__color[0] * 65535) >> 8 
        g8 = int(self.__color[1] * 65535) >> 8
        b8 = int(self.__color[2] * 65535) >> 8
        # print "#%02X%02X%02X" % (r8, g8, b8)
        self.__hexColor = "#%02X%02X%02X" % (r8, g8, b8)

    def getHexColor(self):
        return self.__hexColor

    def getColor(self):
        return self.__color

    def normalizeColor(self):
        if (self.__color[0] > 1) \
            or (self.__color[1] > 1) \
            or (self.__color[2] > 1):

            self.__color[0] = self.__color[0] / 255.0
            self.__color[1] = self.__color[1] / 255.0
            self.__color[2] = self.__color[2] / 255.0


            self.__calcHexColor()

##############################################################################    
#
# Command line interface access to the plugin from PyMOL interpreter
##############################################################################    

def drawcst(selection = "all", 
    gauLogFile = None, 
    objName = "", 
    width = 0.05, 
    relWidth11 = 1.0, 
    relWidth22 = 1.0, 
    relWidth33 = 1.0, 
    relLen11 = 1.0,
    relLen22 = 1.0,
    relLen33 = 1.0,
    color11 = [1.0, 0.0, 0.0], 
    color22 = [0.00 , 1.0 , 0.0], 
    color33 = [0.00 , 0.00 , 1.0],
    pseudo = 1,
    showPseudo = 1,
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

        drawNMRTensors selection, 
            gauLogFile = None, 
            [objName = "", 
            [width=0.02, 
            [relwidth11/22/33 = 1.0,
            [relLen11/22/33 = 1.0,
            [color11 = [1.0, 0.0, 0.0],
            [color22 = [0.0, 1.0, 0.0],
            [color33 = [0.0, 0.0, 1.0],
            [pseudo = 1,
            [showPseudo = 1,
            ]]]]]]]]]

        where:
           "selection"     specifies the atoms for which to draw NMR tensors

           "gauLogFile"    is the name of the Gaussian output file containing 
                           NMR tensors
           
           "objName"       name prefix of the CGOs and pseudoatoms.

           "width"         specifies the width of CGO arrows used for drawing
                           and the size of pseudoatom spheres
           
           "relWidth11/22/33"   relative width of individual components,
                           e.g. relWidth11 = 1.5 makes the arrow 
                           corresponding to 11 component 1.5x thicker relative 
                           to other components

           "relLen11/22/33"     similar to above, only controls the length of 
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

           "showPseudo"    if >0, then the pseudoatoms are shown as spheres at
                           the tips of CGO arrows. 0 makes them hidden

       EXAMPLE
           
           # load an XYZ coordinates of theobromine in standard orientation
           load theobr.xyz, xyz

           # draw the NMR shielding tensor for atom C8, make the CGO representing
           # 11 components 1.5 times thicker and longer, color it black, don't 
           # create pseudoatoms

           drawTensors (xyz and name C8), theobr_NMRTensors.log, relWidth11 = 1.5,
               relLen11 = 1.5, color11 = [0, 0, 0], pseudo=0

    '''

   # sanity check of input values
   # dictionary of value types

    scalarOptionTypes = {
        "objName" : "str",
        "width" : "float",
        "pseudo" : "int",
        "relWidth11" : "float",
        "relWidth22" : "float",
        "relWidth33" : "float",
        "relScale11" : "float",
        "relScale22" : "float",
        "relScale33" : "float",
    }

    listOptionTypes = {
        "color11" : "float",
        "color22" : "float",
        "color33" : "float"
    }


    shieldingTensors = TensorList()

    model = cmd.get_model(selection).atom
    if len(model) == 0:
        print "Missing or invalid atom selection: \"%s\"" % selection
        return

    try:
        shieldingTensors.read(gauLogFile)
    except IOError:
        print "Cannot read Gaussian output \"%s\"!" % gauLogFile
        print "Please check whether this file exists and is valid!"
        return

    if len(shieldingTensors) == 0:
        print "No NMR shielding tensor information present in file \"%s\""\
            % gauLogFile
        print "Please run a Gaussian job with \"NMR(PrintEigenvectors)\" \
           keyword in the route section!"
        return

    for (opt, optType) in scalarOptionTypes.iteritems():
        try:
            exec("%s = %s(%s)" % ( opt, optType, opt))
        except ValueError:
            print "Option \"%s\" must of type \"%s\"! (Was \"%s\")!"\
            % (opt, optType, repr(type(opt)))
            print exc_info()[:1]
            return


    for (opt, optType) in listOptionTypes.iteritems():
        #i = eval("type(%s)" % opt)
        #print "before:", i
        try:
            toList = eval("%s" % opt)
            exec("%s = %s" % (opt, toList))

            #i = eval("type(%s)" % opt)
            #print "after: ", i

            for (i, val) in enumerate(eval("%s" % opt)):
                # print i, val
                exec("%s[%s] = %s(%s)" % (opt, i, optType, val))
                # print "Type: ", eval("type(%s[%s])" % (opt, i))

        except NameError, ValueError:
            print "Option \"%s\" must be list of type \"%s\"!" % (opt, optType)
            return

    # set the relative widths array for emphasizing certain components
    relWidths = [ relWidth11, relWidth22, relWidth33 ]

    # OK, input is tested, now get to work

    for i in model:
        # first double check that the gaussian output contains the 
        # information we need
        try:
            currTensor = shieldingTensors[i.index]
        except KeyError:
            print "Atom %6d has no NMR tensor information in Gaussian \
                output!" % i.index
            continue

        scalingFactors = [
            relScale11,
            relScale22,
            relScale33
        ]

        currTensor.scaleEigVecs(scalingFactors) 
        # now translate the eigenvector to the position of nucleus
        # Gaussian prints the eigenvectors with respect to origin
        currTensor.translateTensor(i.coord)

        # name of the CGO object

        currTensor.setObjName(objName)
        # color of individual components, 11 is red, 22 is green, 33 is blue

        colors = [
            color11,
            color22,
            color33
        ]

        currTensor.setCGOParams(cgoWidth = width, 
            cgoRelWidths = relWidths,
            cgoColors = colors,
        )

        if pseudo != 0:
            currTensor.drawPseudo(True)
        else:
            currTensor.drawPseudo(False)

        if showPseudo != 0:
            currTensor.showPseudo(True)
        else:
            currTensor.showPseudo(False)
            
            
        currTensor.prepareCGOObject(i.coord)

        currTensor.drawCGOObject(cgoObjName)

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
