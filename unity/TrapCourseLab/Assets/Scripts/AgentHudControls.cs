using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

public class AgentHudControls : MonoBehaviour
{
    [SerializeField] private RobotUdpBridge bridge;
    [SerializeField] private bool showHud = true;
    [SerializeField, Range(0f, 1f)] private float dopamine = 0.35f;
    [SerializeField, Range(0f, 1f)] private float norepinephrine = 0.35f;
    [SerializeField, Range(0f, 1f)] private float acetylcholine = 0.35f;
    [SerializeField, Range(0f, 1f)] private float noiseInjection;
    [SerializeField, Range(0f, 1f)] private float calciumGate = 0.45f;
    [SerializeField, Range(0f, 1f)] private float routeExploration = 0.35f;

    private readonly StringBuilder labelBuilder = new StringBuilder(512);
    private Vector2 scrollPosition;

    private void Awake()
    {
        if (bridge == null)
        {
            bridge = GetComponent<RobotUdpBridge>();
        }
    }

    private void Update()
    {
        if (Keyboard.current != null && Keyboard.current.hKey.wasPressedThisFrame)
        {
            showHud = !showHud;
        }

        if (bridge != null)
        {
            bridge.SetHudControls(dopamine, norepinephrine, acetylcholine, noiseInjection, calciumGate, routeExploration);
        }
    }

    private void OnGUI()
    {
        if (bridge == null)
        {
            return;
        }

        if (!showHud)
        {
            GUILayout.BeginArea(new Rect(16f, 16f, 160f, 42f), GUI.skin.box);
            if (GUILayout.Button("Show Agent HUD"))
            {
                showHud = true;
            }
            GUILayout.EndArea();
            return;
        }

        const float panelWidth = 360f;
        GUILayout.BeginArea(new Rect(16f, 16f, panelWidth, Screen.height - 32f), GUI.skin.box);
        GUILayout.BeginHorizontal();
        GUILayout.Label("Agent State");
        if (GUILayout.Button("Hide", GUILayout.Width(72f)))
        {
            showHud = false;
        }
        GUILayout.EndHorizontal();
        scrollPosition = GUILayout.BeginScrollView(scrollPosition, false, true);
        GUILayout.Label(bridge.GetHudText());
        GUILayout.Space(8f);
        DrawSlider("Dopamine", ref dopamine);
        DrawSlider("Norepinephrine", ref norepinephrine);
        DrawSlider("Acetylcholine", ref acetylcholine);
        DrawSlider("Noise Injection", ref noiseInjection);
        DrawSlider("Calcium Gate", ref calciumGate);
        DrawSlider("Route Exploration", ref routeExploration);
        GUILayout.Space(6f);
        GUILayout.Label("H toggles this HUD");
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    private void DrawSlider(string label, ref float value)
    {
        labelBuilder.Clear();
        labelBuilder.Append(label);
        labelBuilder.Append(": ");
        labelBuilder.Append(value.ToString("0.00"));
        GUILayout.Label(labelBuilder.ToString());
        value = GUILayout.HorizontalSlider(value, 0f, 1f);
    }
}
