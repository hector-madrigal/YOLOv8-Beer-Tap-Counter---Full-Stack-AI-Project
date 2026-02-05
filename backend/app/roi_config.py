# Configuracion de ROIs - ORIGINALES que funcionan
ROIS = {
    'ROI_TS': (1194, 26, 1433, 115),
    'ROI_TAP_L': (2020, 749, 83, 198),
    'ROI_TAP_R': (2141, 845, 93, 246),
    'ROI_FLOW_L': (1920, 1101, 172, 483),
    'ROI_FLOW_R': (2055, 1227, 180, 462)
}

def get_roi(roi_name):
    """Obtiene las coordenadas de una ROI por nombre"""
    return ROIS.get(roi_name, (0, 0, 0, 0))

def extract_roi(frame, roi_name):
    """Extrae una region de interes de un frame"""
    x, y, w, h = get_roi(roi_name)
    if w > 0 and h > 0:
        # Validate ROI bounds
        frame_h, frame_w = frame.shape[:2]
        if x + w > frame_w or y + h > frame_h or x < 0 or y < 0:
            print(f"⚠️ ROI {roi_name} out of bounds! ROI: ({x}, {y}, {w}, {h}), Frame: ({frame_w}, {frame_h})")
            return None
        return frame[y:y+h, x:x+w]
    return None
