using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Adjunta este script al dron.
/// Captura frames de la camara del dron y los envia al servidor de
/// deteccion de incendios. Usa la respuesta para orientar el dron.
/// 
/// Setup:
///   1. Asigna la Camera del dron a droneCamera.
///   2. Ajusta serverUrl al IP/puerto donde corre el servidor Python.
///   3. Ajusta captureWidth/captureHeight segun tu resolucion.
///   4. Conecta las funciones de movimiento en OnFireDetected.
/// </summary>
public class FireDetectionClient : MonoBehaviour
{
    [Header("Servidor")]
    [Tooltip("URL base del servidor. Ej: http://192.168.1.100:8000")]
    public string serverUrl = "http://127.0.0.1:8000";

    [Header("Camara del dron")]
    public Camera droneCamera;

    [Header("Configuracion de captura")]
    public int captureWidth  = 640;
    public int captureHeight = 480;

    [Tooltip("Cuantas veces por segundo se manda un frame al servidor")]
    public float captureRate = 5f;   // 5 FPS → menos carga de red

    [Header("Movimiento del dron")]
    [Tooltip("Velocidad maxima de giro hacia el fuego (grados/segundo)")]
    public float yawSpeed     = 60f;
    [Tooltip("Velocidad maxima de inclinacion pitch hacia el fuego")]
    public float pitchSpeed   = 30f;
    [Tooltip("Zona muerta: si el error es menor a este valor no se corrige")]
    public float deadZone     = 0.05f;   // fraccion de la imagen (0-1)

    // ── Estado interno ─────────────────────────────────────────────────
    private RenderTexture _rt;
    private Texture2D     _tex;
    private bool          _busy;          // evita envios solapados
    private float         _timer;

    // Ultima deteccion valida
    private bool  _fireDetected;
    private float _targetCxNorm = 0.5f;   // 0 = izq, 1 = der
    private float _targetCyNorm = 0.5f;   // 0 = arriba, 1 = abajo
    private float _confidence;

    // ── Inicializacion ─────────────────────────────────────────────────
    void Start()
    {
        if (droneCamera == null)
            droneCamera = GetComponentInChildren<Camera>();

        _rt  = new RenderTexture(captureWidth, captureHeight, 24);
        _tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        droneCamera.targetTexture = _rt;

        StartCoroutine(HealthCheck());
    }

    // ── Loop principal ─────────────────────────────────────────────────
    void Update()
    {
        // Enviar frame al ritmo configurado
        _timer += Time.deltaTime;
        if (_timer >= 1f / captureRate && !_busy)
        {
            _timer = 0f;
            StartCoroutine(SendFrame());
        }

        // Orientar el dron segun la ultima deteccion
        if (_fireDetected)
            SteerTowardsFire();
    }

    // ── Captura y envio ────────────────────────────────────────────────
    IEnumerator SendFrame()
    {
        _busy = true;

        // 1. Leer pixels de la RenderTexture
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = _rt;
        _tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        _tex.Apply();
        RenderTexture.active = prev;

        // 2. Codificar como JPEG (mas liviano que PNG para red)
        byte[] jpg = _tex.EncodeToJPG(75);

        // 3. Construir peticion multipart
        WWWForm form = new WWWForm();
        form.AddBinaryData("image", jpg, "frame.jpg", "image/jpeg");

        using UnityWebRequest req = UnityWebRequest.Post(serverUrl + "/detect", form);
        req.timeout = 5;   // segundos

        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            ParseResponse(req.downloadHandler.text);
        }
        else
        {
            Debug.LogWarning($"[FireDetection] Error HTTP: {req.error}");
        }

        _busy = false;
    }

    // ── Parseo de la respuesta JSON ────────────────────────────────────
    void ParseResponse(string json)
    {
        try
        {
            FireResponse resp = JsonUtility.FromJson<FireResponse>(json);
            _fireDetected = resp.fire_detected;

            if (resp.fire_detected && resp.primary_target != null)
            {
                _targetCxNorm = resp.primary_target.cx_norm;
                _targetCyNorm = resp.primary_target.cy_norm;
                _confidence   = resp.primary_target.confidence;

                Debug.Log($"[FireDetection] Fuego en ({_targetCxNorm:F2}, {_targetCyNorm:F2})" +
                          $" conf={_confidence:F2}  ms={resp.inference_ms}");
            }
            else
            {
                Debug.Log("[FireDetection] Sin fuego detectado.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[FireDetection] Error al parsear JSON: {e.Message}\n{json}");
        }
    }

    // ── Control de orientacion ─────────────────────────────────────────
    /// <summary>
    /// Gira el dron para centrar el fuego en la imagen.
    /// cx_norm = 0.5 → fuego centrado horizontalmente.
    /// cy_norm = 0.5 → fuego centrado verticalmente.
    /// </summary>
    void SteerTowardsFire()
    {
        float errorX = _targetCxNorm - 0.5f;   // >0 = derecha, <0 = izquierda
        float errorY = _targetCyNorm - 0.5f;   // >0 = abajo,   <0 = arriba

        // Yaw (giro horizontal)
        if (Mathf.Abs(errorX) > deadZone)
        {
            float yaw = errorX * yawSpeed * Time.deltaTime;
            transform.Rotate(Vector3.up, yaw, Space.World);
        }

        // Pitch (inclinacion vertical) – opcional segun tu dron
        if (Mathf.Abs(errorY) > deadZone)
        {
            float pitch = -errorY * pitchSpeed * Time.deltaTime;
            transform.Rotate(Vector3.right, pitch, Space.Self);
        }

        // TODO: ajusta aqui la velocidad de avance de tu dron
        // Ej: droneRigidbody.AddForce(transform.forward * thrustSpeed);
    }

    // ── Health check al iniciar ────────────────────────────────────────
    IEnumerator HealthCheck()
    {
        using UnityWebRequest req = UnityWebRequest.Get(serverUrl + "/health");
        req.timeout = 3;
        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
            Debug.Log("[FireDetection] Servidor conectado: " + req.downloadHandler.text);
        else
            Debug.LogError("[FireDetection] No se pudo conectar al servidor en " + serverUrl);
    }

    // ── Limpieza ───────────────────────────────────────────────────────
    void OnDestroy()
    {
        if (_rt  != null) _rt.Release();
        if (_tex != null) Destroy(_tex);
    }

    // ── Clases de deserializacion JSON ────────────────────────────────
    [Serializable]
    private class FireResponse
    {
        public bool           fire_detected;
        public int            image_width;
        public int            image_height;
        public Detection[]    detections;
        public Detection      primary_target;
        public float          inference_ms;
    }

    [Serializable]
    private class Detection
    {
        public int   cx;
        public int   cy;
        public float cx_norm;
        public float cy_norm;
        public int   area;
        public int   radius;
        public float confidence;
        public int[] bbox;
    }
}
