import customtkinter
from tkinter import filedialog

from detection_using_cv import *

ROOT_DIR = pathlib.Path(__file__).parent

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title('Image Recognition')

        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.__detect_in_file_button = customtkinter.CTkButton(self, text='Load Image/Video', command=self.__load_image)
        self.__detect_in_file_button.grid(row=0, column=0, padx=10, pady=10)

        self.__detect_using_prim_cam = customtkinter.CTkButton(
            self,
            text='Use Default Camera',
            command=lambda e=None: detect_video('', self.__model, self.__class_labels, use_cam=True, capture_code=0)
        )
        self.__detect_using_prim_cam.grid(row=1, column=0, padx=10, pady=10)

        self.__detect_using_ext_cam = customtkinter.CTkButton(
            self,
            text='Use External Camera',
            command=lambda e=None: detect_video('', self.__model, self.__class_labels, use_cam=True, capture_code=1)
        )
        self.__detect_using_ext_cam.grid(row=2, column=0, padx=10, pady=10)

        self.__config_file = ROOT_DIR / 'model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.__frozen_model_path = ROOT_DIR / 'model/frozen_inference_graph.pb'

        self.__model = cv2.dnn_DetectionModel(str(self.__frozen_model_path), str(self.__config_file))

        self.__label_file_path = ROOT_DIR / 'model/labels.txt'
        with open(self.__label_file_path, 'rt') as fpt:
            self.__class_labels = fpt.read().rstrip('\n').split('\n')

        print(self.__class_labels)
        print(len(self.__class_labels))

    def __load_image(self, *args):
        path = filedialog.askopenfilename(
            title='Select dataset file',
            initialdir=pathlib.Path(__file__).parent.parent,
            filetypes=(("PNG Image", "*.png"), ("JPG Image", "*.jpg"), ("JPEG Image", "*.jpeg"), ("All Files", "*.*"))
        )
        if path == '':
            return
        print(path)
        self.__output_image = detect_image(path, self.__model, self.__class_labels, integrated=False)


if __name__ == '__main__':
    customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

    app = App()
    app.mainloop()