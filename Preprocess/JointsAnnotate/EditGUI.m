function varargout = EditGUI(varargin)
%EDITGUI M-file for EditGUI.fig
%      EDITGUI, by itself, creates a new EDITGUI or raises the existing
%      singleton*.
%
%      H = EDITGUI returns the handle to a new EDITGUI or the handle to
%      the existing singleton*.
%
%      EDITGUI('Property','Value',...) creates a new EDITGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to EditGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      EDITGUI('CALLBACK') and EDITGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in EDITGUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EditGUI

% Last Modified by GUIDE v2.5 23-Aug-2013 18:18:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EditGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @EditGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before EditGUI is made visible.
function EditGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for EditGUI
handles.output = hObject;
handles.ImgExt = 'png';
handles.MatExt = 'gt';
handles.ResultExt = 'gt';

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes EditGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = EditGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function EditImgPath_Callback(hObject, eventdata, handles)
% hObject    handle to EditImgPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of EditImgPath as text
%        str2double(get(hObject,'String')) returns contents of EditImgPath as a double
handles.ImgPath = get(hObject,'String');
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function EditImgPath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to EditImgPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ButtonRun.
function ButtonRun_Callback(hObject, eventdata, handles)
% hObject    handle to ButtonRun (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
contour_edit(handles.ImgPath, handles.ImgExt, handles.MatExt, handles.ResultExt);



function ImgExt_Callback(hObject, eventdata, handles)
% hObject    handle to ImgExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ImgExt as text
%        str2double(get(hObject,'String')) returns contents of ImgExt as a double
handles.ImgExt = get(hObject,'String');
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function ImgExt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ImgExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function MatExt_Callback(hObject, eventdata, handles)
% hObject    handle to MatExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MatExt as text
%        str2double(get(hObject,'String')) returns contents of MatExt as a double
handles.MatExt = get(hObject,'String');
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function MatExt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MatExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ResultExt_Callback(hObject, eventdata, handles)
% hObject    handle to ResultExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ResultExt as text
%        str2double(get(hObject,'String')) returns contents of ResultExt as a double

handles.ResultExt = get(hObject,'String');
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function ResultExt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ResultExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ButtonPath.
function ButtonPath_Callback(hObject, eventdata, handles)
% hObject    handle to ButtonPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.ImgPath = uigetdir;
guidata(hObject,handles);
set(handles.EditImgPath, 'string', handles.ImgPath) ;
