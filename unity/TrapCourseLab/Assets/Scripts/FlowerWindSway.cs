using UnityEngine;

[DisallowMultipleComponent]
public class FlowerWindSway : MonoBehaviour
{
    [SerializeField, Range(0f, 12f)] private float maxAngleDegrees = 3.5f;
    [SerializeField, Range(0.05f, 5f)] private float windSpeed = 1.15f;
    [SerializeField, Range(0f, 1f)] private float crosswind = 0.55f;

    private Transform visualRoot;
    private Vector3 initialLocalPosition;
    private Quaternion initialLocalRotation;
    private Vector3 pivotLocal;
    private float phase;
    private bool initialized;

    private void Awake()
    {
        Initialize();
    }

    private void OnEnable()
    {
        Initialize();
    }

    private void LateUpdate()
    {
        if (!initialized || visualRoot == null || maxAngleDegrees <= 0f)
        {
            return;
        }

        float time = Time.time;
        float primary = Mathf.Sin(time * windSpeed + phase) * maxAngleDegrees;
        float secondary = Mathf.Sin(
            time * windSpeed * 0.73f + phase * 1.37f
        ) * maxAngleDegrees * crosswind;
        Quaternion sway = Quaternion.Euler(secondary, 0f, -primary);

        visualRoot.localPosition = pivotLocal + sway * (initialLocalPosition - pivotLocal);
        visualRoot.localRotation = sway * initialLocalRotation;
    }

    private void OnDisable()
    {
        RestoreInitialPose();
    }

    private void Initialize()
    {
        if (initialized)
        {
            return;
        }

        Renderer[] renderers = GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
        {
            enabled = false;
            return;
        }

        visualRoot = FindDirectVisualRoot(renderers[0].transform);
        initialLocalPosition = visualRoot.localPosition;
        initialLocalRotation = visualRoot.localRotation;

        Bounds combinedBounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            combinedBounds.Encapsulate(renderers[i].bounds);
        }

        Vector3 pivotWorld = new Vector3(
            combinedBounds.center.x,
            combinedBounds.min.y,
            combinedBounds.center.z
        );
        pivotLocal = transform.InverseTransformPoint(pivotWorld);
        phase = transform.position.x * 0.137f + transform.position.z * 0.173f;
        initialized = true;
    }

    private Transform FindDirectVisualRoot(Transform item)
    {
        Transform current = item;
        while (current.parent != null && current.parent != transform)
        {
            current = current.parent;
        }
        return current;
    }

    private void RestoreInitialPose()
    {
        if (!initialized || visualRoot == null)
        {
            return;
        }
        visualRoot.localPosition = initialLocalPosition;
        visualRoot.localRotation = initialLocalRotation;
    }

    private void OnValidate()
    {
        maxAngleDegrees = Mathf.Clamp(maxAngleDegrees, 0f, 12f);
        windSpeed = Mathf.Clamp(windSpeed, 0.05f, 5f);
        crosswind = Mathf.Clamp01(crosswind);
    }
}
