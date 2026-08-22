using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class PGNWCounterbalancedForkTools
{
    private const string RootName = "PGNW Counterbalanced Fork";
    private const float MaximumBindingDistance = 18f;

    private enum Layout
    {
        CalibrationBlueLeft,
        CalibrationYellowLeft,
        ReservedBlueLeft,
        ReservedYellowLeft,
    }

    [MenuItem("Tiny Consciousness/Food/PGNW Fork/Calibration - Blue Left")]
    public static void ApplyCalibrationBlueLeft()
    {
        Apply(Layout.CalibrationBlueLeft);
    }

    [MenuItem("Tiny Consciousness/Food/PGNW Fork/Calibration - Yellow Left")]
    public static void ApplyCalibrationYellowLeft()
    {
        Apply(Layout.CalibrationYellowLeft);
    }

    [MenuItem("Tiny Consciousness/Food/PGNW Fork/Reserved - Blue Left")]
    public static void ApplyReservedBlueLeft()
    {
        Apply(Layout.ReservedBlueLeft);
    }

    [MenuItem("Tiny Consciousness/Food/PGNW Fork/Reserved - Yellow Left")]
    public static void ApplyReservedYellowLeft()
    {
        Apply(Layout.ReservedYellowLeft);
    }

    [MenuItem("Tiny Consciousness/Food/PGNW Fork/Validate Current Layout")]
    public static void ValidateCurrentLayout()
    {
        RobotUdpBridge bridge = UnityEngine.Object.FindAnyObjectByType<RobotUdpBridge>();
        GameObject root = GameObject.Find(RootName);
        if (bridge == null || root == null)
        {
            Debug.LogError("PGNW fork validation requires the robot and a configured fork root.");
            return;
        }

        Dictionary<string, FoodMushroom> foods = FindBoundFoods(root.transform);
        if (!foods.TryGetValue("red", out FoodMushroom red)
            || !foods.TryGetValue("blue", out FoodMushroom blue)
            || !foods.TryGetValue("yellow", out FoodMushroom yellow))
        {
            Debug.LogError("PGNW fork root must contain one red, blue, and yellow pickup.");
            return;
        }

        Vector3 origin = bridge.transform.position;
        float blueDistance = HorizontalDistance(origin, blue.transform.position);
        float yellowDistance = HorizontalDistance(origin, yellow.transform.position);
        float branchDifference = Mathf.Abs(blueDistance - yellowDistance);
        bool valid = branchDifference <= 0.01f
            && HorizontalDistance(origin, red.transform.position) < blueDistance;

        string message =
            $"PGNW fork {(valid ? "valid" : "invalid")}: " +
            $"red={HorizontalDistance(origin, red.transform.position):F3}m, " +
            $"blue={blueDistance:F3}m, yellow={yellowDistance:F3}m, " +
            $"branch_delta={branchDifference:F4}m.";
        if (valid)
        {
            Debug.Log(message);
        }
        else
        {
            Debug.LogError(message);
        }
    }

    private static void Apply(Layout layout)
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("Exit Play mode before changing the PGNW fork layout.");
            return;
        }

        RobotUdpBridge bridge = UnityEngine.Object.FindAnyObjectByType<RobotUdpBridge>();
        Terrain terrain = Terrain.activeTerrain;
        if (bridge == null || terrain == null)
        {
            Debug.LogError("PGNW fork setup requires an active RobotUdpBridge and Terrain.");
            return;
        }

        GameObject root = GameObject.Find(RootName);
        if (root == null)
        {
            root = new GameObject(RootName);
            Undo.RegisterCreatedObjectUndo(root, "Create PGNW counterbalanced fork");
        }

        Dictionary<string, FoodMushroom> foods = FindBoundFoods(root.transform);
        foreach (string feature in new[] { "red", "blue", "yellow" })
        {
            if (foods.ContainsKey(feature))
            {
                continue;
            }

            FoodMushroom nearest = FindNearestUnboundFood(feature, bridge.transform.position);
            if (nearest == null)
            {
                Debug.LogError(
                    $"No standalone {feature} pickup was found within " +
                    $"{MaximumBindingDistance:F0}m of the robot. No layout was applied."
                );
                return;
            }
            Undo.SetTransformParent(
                nearest.transform,
                root.transform,
                "Bind spawn pickup to PGNW fork"
            );
            nearest.name = $"PGNW Fork {char.ToUpperInvariant(feature[0])}{feature.Substring(1)}";
            foods[feature] = nearest;
        }

        Vector3 forward = Vector3.ProjectOnPlane(bridge.transform.forward, Vector3.up).normalized;
        if (forward.sqrMagnitude < 0.5f)
        {
            forward = Vector3.forward;
        }
        Vector3 right = new Vector3(forward.z, 0f, -forward.x);

        bool reserved = layout == Layout.ReservedBlueLeft
            || layout == Layout.ReservedYellowLeft;
        bool blueLeft = layout == Layout.CalibrationBlueLeft
            || layout == Layout.ReservedBlueLeft;

        float redForward = reserved ? 5.5f : 5.0f;
        float branchForward = reserved ? 10.0f : 9.0f;
        float branchLateral = reserved ? 3.8f : 3.5f;
        if (reserved)
        {
            forward = Quaternion.AngleAxis(18f, Vector3.up) * forward;
            right = new Vector3(forward.z, 0f, -forward.x);
        }

        Vector3 origin = bridge.transform.position;
        Vector3 redPosition = origin + forward * redForward;
        Vector3 leftPosition = origin + forward * branchForward - right * branchLateral;
        Vector3 rightPosition = origin + forward * branchForward + right * branchLateral;

        PlaceOnTerrain(foods["red"], redPosition, terrain);
        PlaceOnTerrain(
            foods["blue"],
            blueLeft ? leftPosition : rightPosition,
            terrain
        );
        PlaceOnTerrain(
            foods["yellow"],
            blueLeft ? rightPosition : leftPosition,
            terrain
        );

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Selection.activeGameObject = root;
        Debug.Log(
            $"Applied PGNW counterbalanced fork {layout}. " +
            "Blue and yellow are equidistant mirror assignments. " +
            "Validate, visually inspect accessibility, then save the scene."
        );
        ValidateCurrentLayout();
    }

    private static Dictionary<string, FoodMushroom> FindBoundFoods(Transform root)
    {
        Dictionary<string, FoodMushroom> result = new Dictionary<string, FoodMushroom>();
        foreach (FoodMushroom food in root.GetComponentsInChildren<FoodMushroom>(true))
        {
            string feature = food.ObservableFeature;
            if (feature == "red" || feature == "blue" || feature == "yellow")
            {
                result[feature] = food;
            }
        }
        return result;
    }

    private static FoodMushroom FindNearestUnboundFood(string feature, Vector3 origin)
    {
        FoodMushroom best = null;
        float bestDistance = MaximumBindingDistance;
        foreach (FoodMushroom food in UnityEngine.Object.FindObjectsByType<FoodMushroom>(
            FindObjectsInactive.Exclude
        ))
        {
            if (food.ObservableFeature != feature || IsScattered(food.transform))
            {
                continue;
            }
            float distance = HorizontalDistance(origin, food.transform.position);
            if (distance < bestDistance)
            {
                best = food;
                bestDistance = distance;
            }
        }
        return best;
    }

    private static bool IsScattered(Transform transform)
    {
        for (Transform current = transform; current != null; current = current.parent)
        {
            if (current.name.StartsWith("Scattered ", StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    private static void PlaceOnTerrain(FoodMushroom food, Vector3 target, Terrain terrain)
    {
        Vector3 current = food.transform.position;
        float currentGround = terrain.SampleHeight(current) + terrain.transform.position.y;
        float verticalOffset = Mathf.Clamp(current.y - currentGround, 0.10f, 3.0f);
        target.y = terrain.SampleHeight(target) + terrain.transform.position.y + verticalOffset;
        Undo.RecordObject(food.transform, "Apply PGNW fork position");
        food.transform.position = target;
        EditorUtility.SetDirty(food.transform);
    }

    private static float HorizontalDistance(Vector3 first, Vector3 second)
    {
        float x = first.x - second.x;
        float z = first.z - second.z;
        return Mathf.Sqrt(x * x + z * z);
    }
}
