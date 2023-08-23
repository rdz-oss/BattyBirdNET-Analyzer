import concurrent.futures
import os
import sys
from multiprocessing import freeze_support
import gradio as gr
import webview
import bat_ident
import config as cfg
import segments
import utils
import logging
import librosa
logging.basicConfig(filename='bat_gui.log', encoding='utf-8', level=logging.DEBUG)

_WINDOW: webview.Window


_AREA_ONE = "EU"
_AREA_TWO = "Bavaria"
_AREA_THREE = "USA"
_AREA_FOUR = "Scotland"
_AREA_FIFE = "UK"

#
# MODEL part mixed with CONTROLER
#
OUTPUT_TYPE_MAP = {"Raven selection table": "table", "Audacity": "audacity", "R": "r", "CSV": "csv"}
ORIGINAL_MODEL_PATH = cfg.MODEL_PATH
ORIGINAL_MDATA_MODEL_PATH = cfg.MDATA_MODEL_PATH
ORIGINAL_LABELS_FILE = cfg.LABELS_FILE
ORIGINAL_TRANSLATED_LABELS_PATH = cfg.TRANSLATED_BAT_LABELS_PATH # cfg.TRANSLATED_LABELS_PATH

def analyzeFile_wrapper(entry):
    #return (entry[0], analyze.analyzeFile(entry))
    return (entry[0], bat_ident.analyze_file(entry))
def validate(value, msg):
    """Checks if the value ist not falsy.
    If the value is falsy, an error will be raised.
    Args:
        value: Value to be tested.
        msg: Message in case of an error.
    """
    if not value:
        raise gr.Error(msg)


def runSingleFileAnalysis(input_path,
                          confidence,
                          sensitivity,
                          overlap,
                          species_list_choice,
                          locale):
    validate(input_path, "Please select a file.")
    logging.info('first level')
    return runAnalysis(
        species_list_choice,
        input_path,
        None,
        confidence,
        sensitivity,
        overlap,
        "csv",
        "en" if not locale else locale,
        1,
        4,
        None,
        progress=None,
    )

def runAnalysis(
    species_list_choice: str,
    input_path: str,
    output_path: str | None,
    confidence: float,
    sensitivity: float,
    overlap: float,
    output_type: str,
    locale: str,
    batch_size: int,
    threads: int,
    input_dir: str,
    progress: gr.Progress | None,
):
    """Starts the analysis.
    Args:
        input_path: Either a file or directory.
        output_path: The output path for the result, if None the input_path is used
        confidence: The selected minimum confidence.
        sensitivity: The selected sensitivity.
        overlap: The selected segment overlap.
        species_list_choice: The choice for the species list.
        species_list_file: The selected custom species list file.
        lat: The selected latitude.
        lon: The selected longitude.
        week: The selected week of the year.
        use_yearlong: Use yearlong instead of week.
        sf_thresh: The threshold for the predicted species list.
        custom_classifier_file: Custom classifier to be used.
        output_type: The type of result to be generated.
        locale: The translation to be used.
        batch_size: The number of samples in a batch.
        threads: The number of threads to be used.
        input_dir: The input directory.
        progress: The gradio progress bar.
    """
    logging.info('second level')
    if progress is not None:
        progress(0, desc="Preparing ...")
    # locale = locale.lower()
    # Load eBird codes, labels
    #cfg.CODES = analyze.loadCodes()
    # cfg.LABELS = utils.readLines(ORIGINAL_LABELS_FILE)
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = -1, -1, -1
    cfg.LOCATION_FILTER_THRESHOLD = 0.03
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    cfg.BAT_CLASSIFIER_LOCATION = os.path.join(script_dir, cfg.BAT_CLASSIFIER_LOCATION)

    if species_list_choice == "Bavaria":
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-Bavaria-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-Bavaria-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "de"

    elif species_list_choice == "EU":
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-EU-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-EU-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"

    elif species_list_choice == "Scotland":
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-Scotland-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-Scotland-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"

    elif species_list_choice == "UK":
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-UK-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-UK-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"

    elif species_list_choice == "USA":
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-USA-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-USA-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"

    else:
        cfg.CUSTOM_CLASSIFIER = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-EU-144kHz.tflite"
        cfg.LABELS_FILE = cfg.BAT_CLASSIFIER_LOCATION + "/BattyBirdNET-EU-144kHz_Labels.txt"
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        cfg.LATITUDE = -1
        cfg.LONGITUDE = -1
        cfg.SPECIES_LIST_FILE = None
        cfg.SPECIES_LIST = []
        locale = "en"

    # Load translated labels
    lfile = os.path.join(cfg.TRANSLATED_BAT_LABELS_PATH,
                         os.path.basename(cfg.LABELS_FILE).replace(".txt", f"_{locale}.txt"))
    if not locale in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = utils.readLines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    if len(cfg.SPECIES_LIST) == 0:
        print(f"Species list contains {len(cfg.LABELS)} species")
    else:
        print(f"Species list contains {len(cfg.SPECIES_LIST)} species")

    cfg.INPUT_PATH = input_path

    if input_dir:
        cfg.OUTPUT_PATH = output_path if output_path else input_dir
    else:
        cfg.OUTPUT_PATH = output_path if output_path else input_path.split(".", 1)[0] + ".csv"

    # Parse input files
    if input_dir:
        cfg.FILE_LIST = utils.collect_audio_files(input_dir)
        cfg.INPUT_PATH = input_dir
    elif os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    validate(cfg.FILE_LIST, "No audio files found.")
    cfg.MIN_CONFIDENCE = confidence
    cfg.SIGMOID_SENSITIVITY = sensitivity
    cfg.SIG_OVERLAP = overlap

    # Set result type
    cfg.RESULT_TYPE = OUTPUT_TYPE_MAP[output_type] if output_type in OUTPUT_TYPE_MAP else output_type.lower()

    if not cfg.RESULT_TYPE in ["table", "audacity", "r", "csv"]:
        cfg.RESULT_TYPE = "table"
    # Set number of threads
    if input_dir:
        cfg.CPU_THREADS = max(1, int(threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(threads))
    # Set batch size
    cfg.BATCH_SIZE = max(1, int(batch_size))
    flist = []

    for f in cfg.FILE_LIST:
        flist.append((f, cfg.get_config()))

    result_list = []

    if progress is not None:
        progress(0, desc="Starting ...")
    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in flist:
            result = analyzeFile_wrapper(entry)
            result_list.append(result)
    else:
        executor = None
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(analyzeFile_wrapper, arg) for arg in flist)

            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()
                result_list.append(result)
    return [[os.path.relpath(r[0], input_dir), r[1]] for r in result_list] if input_dir else cfg.OUTPUT_PATH

def extractSegments_wrapper(entry):
    return (entry[0][0], segments.extractSegments(entry))
def extract_segments(audio_dir, result_dir, output_dir, min_conf, num_seq, seq_length, threads, progress=gr.Progress()):
    validate(audio_dir, "No audio directory selected")

    if not result_dir:
        result_dir = audio_dir

    if not output_dir:
        output_dir = audio_dir

    if progress is not None:
        progress(0, desc="Searching files ...")


    # Parse audio and result folders
    cfg.FILE_LIST = segments.parseFolders(audio_dir, result_dir)

    # Set output folder
    cfg.OUTPUT_PATH = output_dir

    # Set number of threads
    cfg.CPU_THREADS = int(threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, min_conf))

    # Parse file list and make list of segments
    cfg.FILE_LIST = segments.parseFiles(cfg.FILE_LIST, max(1, int(num_seq)))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(entry, max(cfg.SIG_LENGTH, float(seq_length)), cfg.get_config()) for entry in cfg.FILE_LIST]

    result_list = []

    # Extract segments
    if cfg.CPU_THREADS < 2:
        for i, entry in enumerate(flist):
            result = extractSegments_wrapper(entry)
            result_list.append(result)

            if progress is not None:
                progress((i, len(flist)), total=len(flist), unit="files")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            futures = (executor.submit(extractSegments_wrapper, arg) for arg in flist)
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress is not None:
                    progress((i, len(flist)), total=len(flist), unit="files")
                result = f.result()

                result_list.append(result)

    return [[os.path.relpath(r[0], audio_dir), r[1]] for r in result_list]


def select_file(filetypes=()):
    """Creates a file selection dialog.
    Args:
        filetypes: List of filetypes to be filtered in the dialog.
    Returns:
        The selected file or None of the dialog was canceled.
    """
    files = _WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=filetypes)
    return files[0] if files else None

def format_seconds(secs: float):
    """Formats a number of seconds into a string.

    Formats the seconds into the format "h:mm:ss.ms"

    Args:
        secs: Number of seconds.

    Returns:
        A string with the formatted seconds.
    """
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    return "{:2.0f}:{:02.0f}:{:06.3f}".format(hours, minutes, secs)

def select_directory(collect_files=True):
    """Shows a directory selection system dialog.

    Uses the pywebview to create a system dialog.

    Args:
        collect_files: If True, also lists a files inside the directory.

    Returns:
        If collect_files==True, returns (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    dir_name = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG)

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name[0])

        return dir_name[0], [
            [os.path.relpath(file, dir_name[0]), format_seconds(librosa.get_duration(filename=file))] for file in files
        ]

    return dir_name[0] if dir_name else None


def show_species_choice(choice: str):
    """Sets the visibility of the species list choices.
    Args:
        choice: The label of the currently active choice.
    Returns:
        A list of [
            Row update,
            File update,
            Column update,
            Column update,
        ]
    """
    return [
        gr.Row.update(visible=True),
        gr.File.update(visible=False),
        gr.Column.update(visible=False),
        gr.Column.update(visible=False),
    ]






#
# VIEW - This is where the UI elements are defined
#

def sample_sliders(opened=True):
    """Creates the gradio accordion for the inference settings.
    Args:
        opened: If True the accordion is open on init.
    Returns:
        A tuple with the created elements:
        (Slider (min confidence), Slider (sensitivity), Slider (overlap))
    """
    with gr.Accordion("Inference settings", open=opened):
        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.01, label="Minimum Confidence", info="Minimum confidence threshold."
            )
            sensitivity_slider = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=1,
                step=0.01,
                label="Sensitivity",
                info="Detection sensitivity; Higher values result in higher sensitivity.",
            )
            overlap_slider = gr.Slider(
                minimum=0, maximum=2.99, value=0, step=0.01, label="Overlap", info="Overlap of prediction segments."
            )

    return confidence_slider, sensitivity_slider, overlap_slider

def locale():
    """Creates the gradio elements for locale selection
    Reads the translated labels inside the checkpoints directory.
    Returns:
        The dropdown element.
    """
    label_files = os.listdir(os.path.join(os.path.dirname(sys.argv[0]), ORIGINAL_TRANSLATED_LABELS_PATH))
    options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

    return gr.Dropdown(options, value="EN", label="Locale", info="Locale for the translated species common names.",visible=False)

def species_lists(opened=True):
    """Creates the gradio accordion for species selection.
    Args:
        opened: If True the accordion is open on init.
    Returns:
        A tuple with the created elements:
        (Radio (choice), File (custom species list), Slider (lat), Slider (lon), Slider (week), Slider (threshold), Checkbox (yearlong?), State (custom classifier))
    """
    with gr.Accordion("Area selection", open=opened):
        with gr.Row():
            species_list_radio = gr.Radio(
                [_AREA_ONE, _AREA_TWO, _AREA_THREE, _AREA_FOUR, _AREA_FIFE],
                value="All regions",
                label="Regions list",
                info="List of all possible regions",
                elem_classes="d-block",
            )
            # species_list_radio.change(
            #     show_species_choice,
            #     inputs=[species_list_radio],
            #     outputs=[ ],
            #     show_progress=False,
            # )
            #
    return species_list_radio

#
# Design main frame for analysis of a single file
#
def build_single_analysis_tab():
    with gr.Tab("Single file"):
        audio_input = gr.Audio(type="filepath", label="file", elem_id="single_file_audio")
        confidence_slider, sensitivity_slider, overlap_slider = sample_sliders(False)
        species_list_radio = species_lists(False)
        locale_radio = locale()

        inputs = [
            audio_input,
            confidence_slider,
            sensitivity_slider,
            overlap_slider,
            species_list_radio,
            locale_radio
        ]

        output_dataframe = gr.Dataframe(
            type="pandas",
            headers=["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"],
            elem_classes="mh-200",
        )
        single_file_analyze = gr.Button("Analyze")
        single_file_analyze.click(runSingleFileAnalysis,
                                  inputs=inputs,
                                  outputs=output_dataframe,
                                  )

def build_segments_tab():
    with gr.Tab("Segments"):
        audio_directory_state = gr.State()
        result_directory_state = gr.State()
        output_directory_state = gr.State()

        def select_directory_to_state_and_tb():
            return (select_directory(collect_files=False),) * 2

        with gr.Row():
            select_audio_directory_btn = gr.Button("Select audio directory (recursive)")
            selected_audio_directory_tb = gr.Textbox(show_label=False, interactive=False)
            select_audio_directory_btn.click(
                select_directory_to_state_and_tb,
                outputs=[selected_audio_directory_tb, audio_directory_state],
                show_progress=False,
            )

        with gr.Row():
            select_result_directory_btn = gr.Button("Select result directory")
            selected_result_directory_tb = gr.Textbox(
                show_label=False, interactive=False, placeholder="Same as audio directory if not selected"
            )
            select_result_directory_btn.click(
                select_directory_to_state_and_tb,
                outputs=[result_directory_state, selected_result_directory_tb],
                show_progress=False,
            )

        with gr.Row():
            select_output_directory_btn = gr.Button("Select output directory")
            selected_output_directory_tb = gr.Textbox(
                show_label=False, interactive=False, placeholder="Same as audio directory if not selected"
            )
            select_output_directory_btn.click(
                select_directory_to_state_and_tb,
                outputs=[selected_output_directory_tb, output_directory_state],
                show_progress=False,
            )

        min_conf_slider = gr.Slider(
            minimum=0.1, maximum=0.99, step=0.01, label="Minimum confidence", info="Minimum confidence threshold."
        )
        num_seq_number = gr.Number(
            100, label="Max number of segments", info="Maximum number of randomly extracted segments per species."
        )
        seq_length_number = gr.Number(3.0, label="Sequence length", info="Length of extracted segments in seconds.")
        threads_number = gr.Number(4, label="Threads", info="Number of CPU threads.")

        extract_segments_btn = gr.Button("Extract segments")

        result_grid = gr.Matrix(headers=["File", "Execution"], elem_classes="mh-200")

        extract_segments_btn.click(
            extract_segments,
            inputs=[
                audio_directory_state,
                result_directory_state,
                output_directory_state,
                min_conf_slider,
                num_seq_number,
                seq_length_number,
                threads_number,
            ],
            outputs=result_grid,
        )

if __name__ == "__main__":
    freeze_support()
    with gr.Blocks(
        css=r".d-block .wrap {display: block !important;} .mh-200 {max-height: 300px; overflow-y: auto !important;} footer {display: none !important;} #single_file_audio, #single_file_audio * {max-height: 81.6px; min-height: 0;}",
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_single_analysis_tab()
        build_segments_tab()

    url = demo.queue(api_open=False).launch(prevent_thread_lock=True, quiet=True)[1]
    _WINDOW = webview.create_window("BattyBirdNET-Analyzer", url.rstrip("/") +
                                    "?__theme=light", min_size=(1024, 768))
    webview.start(private_mode=False)
