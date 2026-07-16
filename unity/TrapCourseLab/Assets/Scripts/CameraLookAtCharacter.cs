using UnityEngine;

public class CameraLookAtCharacter : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private Vector3 targetOffset = new Vector3(0f, 0.3f, 0f);
    [SerializeField] private float distance = 4f;
    [SerializeField] private float height = 0.8f;
    [SerializeField] private float followSharpness = 10f;
    [SerializeField] private bool findTargetOnStart = true;
    [SerializeField] private bool useTargetForward = false;
    [SerializeField] private bool autoFrameTarget = true;
    [SerializeField] private float framingPadding = 1.35f;
    [SerializeField] private float minimumFrameDistance = 0.35f;
    [SerializeField] private bool useStableRootLookPoint = true;
    [SerializeField] private float lookPointSharpness = 6f;
    [SerializeField] private bool lockFramingSizeOnStart = true;

    private Bounds targetBounds;
    private float nextTargetSearchTime;
    private Camera targetCamera;
    private Vector3 smoothedLookPoint;
    private bool hasSmoothedLookPoint;
    private float stableLookHeight = 1f;
    private float stableFrameDistance;
    private float stableFrameHeight;
    private bool hasStableFrame;

    private void Start()
    {
        targetCamera = GetComponent<Camera>();
        if (targetCamera != null)
        {
            targetCamera.nearClipPlane = 0.01f;
        }

        if (target == null && findTargetOnStart)
        {
            target = FindSceneCharacter();
        }

        if (target == null)
        {
            Debug.LogWarning("CameraLookAtCharacter could not find a character in the scene. Drag your FBX from the Project window into the Hierarchy, or assign the Target field on Main Camera.");
            return;
        }

        RefreshBounds();
        CaptureStableFrame();
        ResetSmoothedLookPoint();
        SnapToTarget();
        LogTargetInfo();
    }

    private void LateUpdate()
    {
        if (target == null)
        {
            TryFindTargetAgain();
            return;
        }

        RefreshBounds();

        Vector3 lookPoint = GetSmoothedLookPoint();
        Vector3 desiredPosition = GetCameraPosition(lookPoint);

        float t = 1f - Mathf.Exp(-followSharpness * Time.deltaTime);
        transform.position = Vector3.Lerp(transform.position, desiredPosition, t);
        transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(lookPoint - transform.position), t);
    }

    [ContextMenu("Snap To Target")]
    private void SnapToTarget()
    {
        if (target == null)
        {
            return;
        }

        Vector3 lookPoint = GetRawLookPoint();
        transform.position = GetCameraPosition(lookPoint);
        transform.rotation = Quaternion.LookRotation(lookPoint - transform.position);
    }

    public void SetTargetAndSnap(Transform newTarget)
    {
        if (newTarget == null)
        {
            return;
        }
        target = newTarget;
        RefreshBounds();
        CaptureStableFrame();
        ResetSmoothedLookPoint();
        SnapToTarget();
    }

    private Vector3 GetCameraPosition(Vector3 lookPoint)
    {
        Vector3 viewDirection = useTargetForward ? -target.forward : Vector3.back;
        float frameDistance = GetCameraDistance();
        float frameHeight = GetCameraHeight();
        return lookPoint + viewDirection.normalized * frameDistance + Vector3.up * frameHeight;
    }

    private float GetCameraDistance()
    {
        if (!autoFrameTarget)
        {
            return distance;
        }

        if (lockFramingSizeOnStart && hasStableFrame)
        {
            return stableFrameDistance;
        }

        return GetFramingDistance();
    }

    private float GetCameraHeight()
    {
        if (!autoFrameTarget)
        {
            return height;
        }

        if (lockFramingSizeOnStart && hasStableFrame)
        {
            return stableFrameHeight;
        }

        return Mathf.Max(targetBounds.extents.y * 0.15f, height * 0.1f);
    }

    private float GetFramingDistance()
    {
        if (targetCamera == null || targetBounds.size == Vector3.zero)
        {
            return distance;
        }

        float verticalFov = targetCamera.fieldOfView * Mathf.Deg2Rad;
        float horizontalFov = 2f * Mathf.Atan(Mathf.Tan(verticalFov * 0.5f) * targetCamera.aspect);
        float verticalDistance = targetBounds.extents.y / Mathf.Tan(verticalFov * 0.5f);
        float horizontalDistance = targetBounds.extents.x / Mathf.Tan(horizontalFov * 0.5f);
        float depthPadding = targetBounds.extents.z;

        return Mathf.Max(minimumFrameDistance, (Mathf.Max(verticalDistance, horizontalDistance) + depthPadding) * framingPadding);
    }

    private Transform FindSceneCharacter()
    {
        Animator animator = FindAnyObjectByType<Animator>();
        if (animator != null)
        {
            return animator.transform;
        }

        SkinnedMeshRenderer skinnedMesh = FindAnyObjectByType<SkinnedMeshRenderer>();
        if (skinnedMesh != null)
        {
            return skinnedMesh.transform;
        }

        return null;
    }

    private void TryFindTargetAgain()
    {
        if (!findTargetOnStart || Time.time < nextTargetSearchTime)
        {
            return;
        }

        nextTargetSearchTime = Time.time + 0.5f;
        target = FindSceneCharacter();

        if (target != null)
        {
            RefreshBounds();
            CaptureStableFrame();
            ResetSmoothedLookPoint();
            SnapToTarget();
            LogTargetInfo();
        }
    }

    private void LogTargetInfo()
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
        int enabledRenderers = 0;

        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i].enabled && renderers[i].gameObject.activeInHierarchy)
            {
                enabledRenderers++;
            }
        }

        Debug.Log($"CameraLookAtCharacter target '{target.name}' has {renderers.Length} child renderers, {enabledRenderers} enabled. Bounds center {targetBounds.center}, size {targetBounds.size}. Camera position {transform.position}.");
    }

    private void RefreshBounds()
    {
        if (target == null)
        {
            return;
        }

        Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
        bool found = false;
        Bounds visibleBounds = new Bounds();
        foreach (Renderer item in renderers)
        {
            if (!IsFramingRenderer(item))
            {
                continue;
            }
            if (!found)
            {
                visibleBounds = item.bounds;
                found = true;
            }
            else
            {
                visibleBounds.Encapsulate(item.bounds);
            }
        }
        if (!found)
        {
            targetBounds = new Bounds(target.position + targetOffset, Vector3.one);
            return;
        }
        targetBounds = visibleBounds;
    }

    private static bool IsFramingRenderer(Renderer item)
    {
        if (item == null || !item.enabled || !item.gameObject.activeInHierarchy)
        {
            return false;
        }
        return item is SkinnedMeshRenderer || item is MeshRenderer;
    }

    private Vector3 GetSmoothedLookPoint()
    {
        Vector3 rawLookPoint = GetRawLookPoint();
        if (!hasSmoothedLookPoint)
        {
            smoothedLookPoint = rawLookPoint;
            hasSmoothedLookPoint = true;
            return smoothedLookPoint;
        }

        float t = 1f - Mathf.Exp(-lookPointSharpness * Time.deltaTime);
        smoothedLookPoint = Vector3.Lerp(smoothedLookPoint, rawLookPoint, t);
        return smoothedLookPoint;
    }

    private void ResetSmoothedLookPoint()
    {
        smoothedLookPoint = GetRawLookPoint();
        hasSmoothedLookPoint = true;
    }

    private void CaptureStableFrame()
    {
        if (!lockFramingSizeOnStart || targetBounds.size == Vector3.zero)
        {
            return;
        }

        stableLookHeight = Mathf.Max(0.1f, targetBounds.extents.y);
        stableFrameDistance = autoFrameTarget ? GetFramingDistance() : distance;
        stableFrameHeight = autoFrameTarget ? Mathf.Max(targetBounds.extents.y * 0.15f, height * 0.1f) : height;
        hasStableFrame = true;
    }

    private Vector3 GetRawLookPoint()
    {
        if (useStableRootLookPoint || targetBounds.size == Vector3.zero)
        {
            return target.position + Vector3.up * stableLookHeight + targetOffset;
        }

        return targetBounds.center + targetOffset;
    }
}
