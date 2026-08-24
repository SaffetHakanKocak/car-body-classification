import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.model import create_model


MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.json"
EVALUATION_DIR = ROOT_DIR / "outputs_b2_v4"
TEST_METRICS_PATH = EVALUATION_DIR / "test_metrics.json"
TEST_CLASSIFICATION_REPORT_PATH = EVALUATION_DIR / "test_classification_report.txt"

IMAGE_SIZE = 224
MODEL_NAME = "efficientnet_b2"
DROPOUT_RATE = 0.4
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FINAL_TRAINING_LOG = """Epoch 27/30 | LR: 0.000013 | Train loss: 0.2749 | Val loss: 0.3089 | Train acc: 0.9994 | Val acc: 0.9824 | Val macro F1: 0.9823
No improvement. Patience counter: 5/5
Early stopping triggered.
Eğitim tamamlandı.
En iyi validation macro F1: 0.9861681552719999"""

FINAL_TRAINING_SUMMARY = {
    "Final Epoch": "27/30",
    "Best Val Macro F1": 0.9861681552719999,
    "Final Val Macro F1": 0.9823,
    "Final Val Accuracy": 0.9824,
    "Final Train Accuracy": 0.9994,
    "Final Train Loss": 0.2749,
    "Final Val Loss": 0.3089,
}


def format_percent(value):
    return f"{value * 100:.2f}%"


def show_compact_metrics(metrics):
    metric_columns = st.columns(3)
    for index, (name, value) in enumerate(metrics):
        metric_columns[index % 3].markdown(
            (
                "<div style='font-size: 0.86rem; line-height: 1.45; margin-bottom: 0.75rem;'>"
                f"<span style='opacity: 0.78;'>{name}</span><br>"
                f"<strong>{value}</strong>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResizeWithPadding:
    def __init__(self, size=224, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        image = image.convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - image.width) // 2
        top = (self.size - image.height) // 2
        new_image.paste(image, (left, top))
        return new_image


def get_transform():
    return transforms.Compose(
        [
            ResizeWithPadding(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != 8:
        raise ValueError("class_names.json must contain 8 classes.")

    device = get_device()
    model = create_model(
        num_classes=len(class_names),
        dropout_rate=DROPOUT_RATE,
        pretrained=False,
        model_name=MODEL_NAME,
    )
    state_dict = load_state_dict(MODEL_PATH, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(image, model, class_names, device):
    transform = get_transform()
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities_tensor = torch.softmax(outputs, dim=1).squeeze(0).cpu()

    predicted_index = int(probabilities_tensor.argmax().item())
    confidence = float(probabilities_tensor[predicted_index].item())
    probabilities = {
        class_name: float(probabilities_tensor[index].item())
        for index, class_name in enumerate(class_names)
    }

    return {
        "predicted_index_0_based": predicted_index,
        "predicted_label_1_based": predicted_index + 1,
        "predicted_class": class_names[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def is_supported_image(path):
    return path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_FORMATS


def safe_extract_zip(uploaded_zip, target_dir):
    target_dir = Path(target_dir).resolve()

    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as archive:
        for member in archive.infolist():
            member_path = (target_dir / member.filename).resolve()
            try:
                member_path.relative_to(target_dir)
            except ValueError as error:
                raise ValueError("ZIP file contains an unsafe file path.") from error

        archive.extractall(target_dir)


def find_testdata_root(extract_dir):
    extract_dir = Path(extract_dir)
    testdata_dir = extract_dir / "testdata"

    if testdata_dir.exists() and testdata_dir.is_dir():
        return testdata_dir

    child_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(child_dirs) == 1:
        return child_dirs[0]

    return extract_dir


def predict_directory(testdata_dir, model, class_names, device):
    image_paths = sorted(path for path in Path(testdata_dir).rglob("*") if is_supported_image(path))
    rows = []
    lines = []

    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            result = predict_image(image, model, class_names, device)

        file_name = image_path.name
        predicted_label = result["predicted_label_1_based"]
        lines.append(f"{file_name} | Pred: {predicted_label}")
        rows.append(
            {
                "Image Bytes": image_bytes,
                "Filename": file_name,
                "Pred": predicted_label,
                "Class": result["predicted_class"],
                "Confidence (%)": result["confidence"] * 100,
            }
        )

    return rows, "\n".join(lines)


def parse_prediction_text(text):
    values = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(" | ")
        if len(parts) != 2 or ":" not in parts[1]:
            continue

        file_name = Path(parts[0].strip()).name
        label = int(parts[1].split(":", 1)[1].strip())
        values[file_name] = label

    return values


def compute_batch_metrics(prediction_text, true_text, class_names):
    predictions = parse_prediction_text(prediction_text)
    true_values = parse_prediction_text(true_text)
    common_files = sorted(set(predictions) & set(true_values))

    if not common_files:
        raise ValueError("True.txt ile Preds.txt arasında ortak dosya adı bulunamadı.")

    y_true = [true_values[file_name] for file_name in common_files]
    y_pred = [predictions[file_name] for file_name in common_files]
    labels = list(range(1, len(class_names) + 1))

    metrics = {
        "Common files": len(common_files),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"True {index}" for index in labels],
        columns=[f"Pred {index}" for index in labels],
    )
    normalized_matrix_df = pd.DataFrame(
        normalized_matrix,
        index=[f"True {index}" for index in labels],
        columns=[f"Pred {index}" for index in labels],
    )

    return metrics, matrix_df, normalized_matrix_df


def create_confusion_heatmap(matrix_df):
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=axis,
    )
    axis.set_xlabel("Tahmin edilen sınıf")
    axis.set_ylabel("Gerçek sınıf")
    axis.set_title("Normalize Edilmiş Karışıklık Matrisi")
    figure.tight_layout()
    return figure


def show_model_info():
    st.sidebar.header("Model Bilgisi")
    st.sidebar.write("Model: EfficientNet-B2")
    st.sidebar.write("En İyi Val Macro F1: 98.62%")
    st.sidebar.write("Son Val Accuracy: 98.24%")
    st.sidebar.write("Test Accuracy: 96.98%")
    st.sidebar.write("Test Macro F1: 97.00%")
    st.sidebar.write("Model Boyutu: 29.86 MB")
    st.sidebar.write("Sınıf Sayısı: 8")


def show_prediction_result(result):
    st.subheader("Tahmin Sonucu")
    st.metric("Tahmin Edilen Sınıf", result["predicted_class"])
    st.metric("Sınıf Numarası", result["predicted_label_1_based"])
    st.metric("Güven Skoru", f"{result['confidence'] * 100:.2f}%")

    probability_rows = [
        {
            "Sınıf": class_name,
            "Olasılık (%)": probability * 100,
        }
        for class_name, probability in result["probabilities"].items()
    ]
    probability_df = pd.DataFrame(probability_rows)

    st.subheader("Sınıf Olasılıkları")
    st.bar_chart(probability_df.set_index("Sınıf"))
    st.dataframe(probability_df, use_container_width=True, hide_index=True)


def show_batch_gallery(prediction_rows, true_labels=None, class_names=None):
    st.subheader("Test Görselleri")

    columns = st.columns(4)
    for index, row in enumerate(prediction_rows):
        caption_lines = [
            row["Filename"],
            f"Tahmin: {row['Pred']} - {row['Class']}",
        ]

        if true_labels is not None and class_names is not None:
            true_label = true_labels.get(row["Filename"])
            if true_label is not None:
                if 1 <= true_label <= len(class_names):
                    true_class = class_names[true_label - 1]
                else:
                    true_class = "Bilinmiyor"

                caption_lines.append(f"Gerçek: {true_label} - {true_class}")
                if true_label == row["Pred"]:
                    caption_lines.append("Sonuç: Doğru")
                else:
                    caption_lines.append("Sonuç: Yanlış")

        with columns[index % 4]:
            st.image(
                row["Image Bytes"],
                caption="\n".join(caption_lines),
                use_container_width=True,
            )


def show_single_prediction(model, class_names, device):
    uploaded_file = st.file_uploader(
        "Araç görseli yükle",
        type=SUPPORTED_FORMATS,
    )

    if uploaded_file is None:
        st.info("Tahmin yapmak için bir görsel yükleyin.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Görsel açılamadı. Lütfen geçerli bir görsel dosyası yükleyin.")
        return

    image_col, result_col = st.columns([1, 1])

    with image_col:
        st.subheader("Yüklenen Görsel")
        st.image(image, use_container_width=True)

    with result_col:
        if st.button("Tahmin Yap", type="primary"):
            try:
                result = predict_image(image, model, class_names, device)
                show_prediction_result(result)
            except Exception as error:
                st.error(f"Tahmin başarısız oldu: {error}")


def show_batch_test(model, class_names, device):
    testdata_zip = st.file_uploader(
        "testdata.zip",
        type=["zip"],
        key="testdata_zip",
    )
    true_file = st.file_uploader(
        "True.txt",
        type=["txt"],
        key="true_file",
    )

    if testdata_zip is None:
        st.info("Toplu tahmin yapmak için testdata.zip dosyasını yükleyin.")
        return

    if st.button("Toplu Testi Çalıştır", type="primary"):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_dir = Path(temp_dir) / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(testdata_zip, extract_dir)
                testdata_dir = find_testdata_root(extract_dir)
                prediction_rows, prediction_text = predict_directory(
                    testdata_dir,
                    model,
                    class_names,
                    device,
                )

            if not prediction_rows:
                st.warning("ZIP içinde desteklenen formatta görsel bulunamadı.")
                return

            st.success(f"{len(prediction_rows)} görsel için tahmin üretildi.")
            true_labels = None
            true_text = None

            if true_file is not None:
                true_text = true_file.getvalue().decode("utf-8")
                true_labels = parse_prediction_text(true_text)

            show_batch_gallery(
                prediction_rows,
                true_labels=true_labels,
                class_names=class_names,
            )

            prediction_df = pd.DataFrame(
                [
                    {key: value for key, value in row.items() if key != "Image Bytes"}
                    for row in prediction_rows
                ]
            )

            if true_labels is not None:
                prediction_df["True"] = prediction_df["Filename"].map(true_labels)
                prediction_df["True Class"] = prediction_df["True"].apply(
                    lambda label: class_names[int(label) - 1]
                    if pd.notna(label) and 1 <= int(label) <= len(class_names)
                    else None
                )
                prediction_df["Result"] = prediction_df.apply(
                    lambda row: "Doğru"
                    if pd.notna(row["True"]) and int(row["True"]) == int(row["Pred"])
                    else "Yanlış"
                    if pd.notna(row["True"])
                    else None,
                    axis=1,
                )

            st.dataframe(prediction_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Preds.txt İndir",
                data=prediction_text.encode("utf-8"),
                file_name="Preds.txt",
                mime="text/plain",
            )
            st.code(prediction_text, language="text")

            if true_text is not None:
                metrics, matrix_df, normalized_matrix_df = compute_batch_metrics(
                    prediction_text,
                    true_text,
                    class_names,
                )
                metric_cols = st.columns(5)
                for column, (name, value) in zip(metric_cols, metrics.items()):
                    if isinstance(value, int):
                        column.metric(name, value)
                    else:
                        column.metric(name, f"{value:.4f}")

                st.subheader("Normalize Edilmiş Karışıklık Matrisi")
                heatmap_figure = create_confusion_heatmap(normalized_matrix_df)
                st.pyplot(heatmap_figure)
                plt.close(heatmap_figure)

                st.subheader("Karışıklık Matrisi Sayıları")
                st.dataframe(matrix_df, use_container_width=True)
        except Exception as error:
            st.error(f"Toplu test başarısız oldu: {error}")


def show_evaluation_graphs():
    metric_values = {
        "Accuracy": 0.9698492462311558,
        "Macro Precision": 0.9709954693856788,
        "Macro Recall": 0.9698958333333333,
        "Macro F1": 0.9699691195571378,
    }

    if TEST_METRICS_PATH.exists():
        with TEST_METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        metric_values = {
            "Accuracy": metrics.get("accuracy", metric_values["Accuracy"]),
            "Macro Precision": metrics.get("macro_precision", metric_values["Macro Precision"]),
            "Macro Recall": metrics.get("macro_recall", metric_values["Macro Recall"]),
            "Macro F1": metrics.get("macro_f1", metric_values["Macro F1"]),
        }

    st.subheader("Öne Çıkan Sonuçlar")
    primary_metrics = [
        ("En İyi Val Macro F1", format_percent(FINAL_TRAINING_SUMMARY["Best Val Macro F1"])),
        ("Son Val Accuracy", format_percent(FINAL_TRAINING_SUMMARY["Final Val Accuracy"])),
        ("Test Accuracy", format_percent(metric_values["Accuracy"])),
        ("Test Macro F1", format_percent(metric_values["Macro F1"])),
    ]

    primary_cols = st.columns(4)
    for column, (name, value) in zip(primary_cols, primary_metrics):
        column.metric(name, value)

    st.subheader("Eğitim Detayları")
    training_details = [
        ("Final Epoch", FINAL_TRAINING_SUMMARY["Final Epoch"]),
        ("Final Val Macro F1", f"{FINAL_TRAINING_SUMMARY['Final Val Macro F1']:.4f}"),
        ("Final Train Accuracy", f"{FINAL_TRAINING_SUMMARY['Final Train Accuracy']:.4f}"),
        ("Final Train Loss", f"{FINAL_TRAINING_SUMMARY['Final Train Loss']:.4f}"),
        ("Final Val Loss", f"{FINAL_TRAINING_SUMMARY['Final Val Loss']:.4f}"),
    ]
    show_compact_metrics(training_details)

    with st.expander("Eğitim Logu"):
        st.code(FINAL_TRAINING_LOG, language="text")

    st.subheader("Test Metrikleri")
    test_details = [(name, f"{value:.4f}") for name, value in metric_values.items()]
    show_compact_metrics(test_details)

    st.subheader("Sınıf Bazlı Test Raporu")
    if TEST_CLASSIFICATION_REPORT_PATH.exists():
        report_text = TEST_CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8")
        st.code(report_text, language="text")
    else:
        st.warning("test_classification_report.txt bulunamadı.")

    loss_path = EVALUATION_DIR / "loss_curve.png"
    accuracy_path = EVALUATION_DIR / "accuracy_curve.png"
    confusion_path = EVALUATION_DIR / "normalized_confusion_matrix.png"

    st.subheader("Eğitim Grafikleri")
    curve_col_1, curve_col_2 = st.columns(2)
    with curve_col_1:
        if loss_path.exists():
            st.image(str(loss_path), caption="Eğitim ve Doğrulama Loss", use_container_width=True)
        else:
            st.warning("loss_curve.png bulunamadı.")

    with curve_col_2:
        if accuracy_path.exists():
            st.image(str(accuracy_path), caption="Eğitim ve Doğrulama Accuracy", use_container_width=True)
        else:
            st.warning("accuracy_curve.png bulunamadı.")

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption="Normalize Edilmiş Karışıklık Matrisi",
            use_container_width=True,
        )
    else:
        st.warning("normalized_confusion_matrix.png bulunamadı.")


def main():
    st.set_page_config(
        page_title="Araba Gövde Tipi Sınıflandırma",
        layout="wide",
    )

    st.title("Araba Gövde Tipi Sınıflandırma")
    st.write(
        "EfficientNet-B2 tabanlı 8 sınıflı araç gövde tipi sınıflandırma sistemi."
    )

    show_model_info()

    try:
        model, class_names, device = load_model_and_classes()
    except Exception as error:
        st.error(str(error))
        st.stop()

    st.sidebar.write(f"Cihaz: {device}")

    single_tab, batch_tab, evaluation_tab = st.tabs(
        ["Tek Görsel", "Toplu Test", "Model Değerlendirme"]
    )

    with single_tab:
        show_single_prediction(model, class_names, device)

    with batch_tab:
        show_batch_test(model, class_names, device)

    with evaluation_tab:
        show_evaluation_graphs()


if __name__ == "__main__":
    main()
