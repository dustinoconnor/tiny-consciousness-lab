using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RedMushroomPrefabTools
{
    private const string SourcePath = "Assets/mushroomRed.glb";
    private const string PrefabPath = "Assets/RedFoodMushroom.prefab";
    private const string ScatterRootName = "Scattered Red Food Mushrooms";
    private const int ScatterTargetCount = 100;
    private const int ScatterSeed = 20260801;
    private const int MaxPlacementAttempts = 20000;
    private const float ExistingFoodSpacing = 14f;
    private const float RedFoodSpacing = 24f;
    private const float VerticalOffset = 1.5f;

    [MenuItem("Tiny Consciousness/Food/Create Red Food Prefab")]
    private static void CreateRedFoodPrefab()
    {
        GameObject source = AssetDatabase.LoadAssetAtPath<GameObject>(SourcePath);
        if (source == null)
        {
            Debug.LogError($"Red mushroom source was not found at {SourcePath}.");
            return;
        }

        GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(source);
        if (instance == null)
        {
            Debug.LogError("Could not instantiate the imported red mushroom.");
            return;
        }

        try
        {
            PrefabUtility.UnpackPrefabInstance(
                instance,
                PrefabUnpackMode.Completely,
                InteractionMode.AutomatedAction
            );
            instance.name = "RedFoodMushroom";
            FoodMushroom food = instance.GetComponent<FoodMushroom>();
            if (food == null)
            {
                food = instance.AddComponent<FoodMushroom>();
            }
            food.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Red);
            BoxCollider pickupCollider = instance.GetComponent<BoxCollider>();
            pickupCollider.isTrigger = true;
            Rigidbody body = instance.GetComponent<Rigidbody>();
            body.isKinematic = true;
            body.useGravity = false;

            bool saved;
            PrefabUtility.SaveAsPrefabAsset(instance, PrefabPath, out saved);
            if (!saved)
            {
                Debug.LogError($"Unity could not save {PrefabPath}.");
                return;
            }

            AssetDatabase.SaveAssets();
            Selection.activeObject = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
            Debug.Log(
                $"Created {PrefabPath} with explicit red observable metadata. " +
                "Inspect its scale and collider before placing it in the terrain."
            );
        }
        finally
        {
            Object.DestroyImmediate(instance);
        }
    }

    [MenuItem("Tiny Consciousness/Food/Repair and Expand to 100 Red Food Mushrooms")]
    private static void ScatterRedFoodMushrooms()
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("Exit Play mode before scattering red mushrooms.");
            return;
        }

        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        if (prefab == null)
        {
            Debug.LogError(
                $"Red food prefab was not found at {PrefabPath}. " +
                "Run Tiny Consciousness/Food/Create Red Food Prefab first."
            );
            return;
        }
        GameObject root = GameObject.Find(ScatterRootName);

        Terrain terrain = Terrain.activeTerrain;
        if (terrain == null)
        {
            Debug.LogError("No active Terrain was found for red-mushroom scattering.");
            return;
        }

        FoodMushroom[] foods = UnityEngine.Object.FindObjectsByType<FoodMushroom>(
            FindObjectsInactive.Exclude,
            FindObjectsSortMode.None
        );
        List<Vector3> existingPositions = new List<Vector3>();
        bool hasBlueBounds = false;
        float minX = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float minZ = float.PositiveInfinity;
        float maxZ = float.NegativeInfinity;
        foreach (FoodMushroom food in foods)
        {
            if (food == null)
            {
                continue;
            }
            Vector3 position = food.transform.position;
            existingPositions.Add(position);
            if (food.ObservableFeature != "blue")
            {
                continue;
            }
            hasBlueBounds = true;
            minX = Mathf.Min(minX, position.x);
            maxX = Mathf.Max(maxX, position.x);
            minZ = Mathf.Min(minZ, position.z);
            maxZ = Mathf.Max(maxZ, position.z);
        }
        if (!hasBlueBounds || maxX - minX < 1f || maxZ - minZ < 1f)
        {
            Debug.LogError(
                "At least two spatially distributed blue FoodMushrooms are " +
                "required to define the occupied terrain region."
            );
            return;
        }

        Vector3 terrainOrigin = terrain.transform.position;
        Vector3 terrainSize = terrain.terrainData.size;
        minX = Mathf.Max(minX, terrainOrigin.x);
        maxX = Mathf.Min(maxX, terrainOrigin.x + terrainSize.x);
        minZ = Mathf.Max(minZ, terrainOrigin.z);
        maxZ = Mathf.Min(maxZ, terrainOrigin.z + terrainSize.z);

        List<Vector3> redPositions = new List<Vector3>();
        if (root == null)
        {
            root = new GameObject(ScatterRootName);
            Undo.RegisterCreatedObjectUndo(root, "Scatter red food mushrooms");
        }
        else
        {
            FoodMushroom[] existingReds = root.GetComponentsInChildren<FoodMushroom>(true);
            foreach (FoodMushroom existingRed in existingReds)
            {
                Vector3 repaired = existingRed.transform.position;
                repaired.y = terrain.SampleHeight(repaired) + terrainOrigin.y + VerticalOffset;
                Undo.RecordObject(existingRed.transform, "Lift red food mushroom");
                existingRed.transform.position = repaired;
                Undo.RecordObject(existingRed, "Configure red food mushroom");
                existingRed.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Red);
                EditorUtility.SetDirty(existingRed);
                redPositions.Add(repaired);
            }
        }

        System.Random random = new System.Random(ScatterSeed);
        int attempts = 0;
        while (redPositions.Count < ScatterTargetCount && attempts < MaxPlacementAttempts)
        {
            attempts += 1;
            float x = Mathf.Lerp(minX, maxX, (float)random.NextDouble());
            float z = Mathf.Lerp(minZ, maxZ, (float)random.NextDouble());
            Vector3 candidate = new Vector3(x, 0f, z);
            if (!IsSeparated(candidate, existingPositions, ExistingFoodSpacing) ||
                !IsSeparated(candidate, redPositions, RedFoodSpacing))
            {
                continue;
            }

            candidate.y = terrain.SampleHeight(candidate) + terrainOrigin.y + VerticalOffset;
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(
                prefab,
                root.transform
            );
            if (instance == null)
            {
                continue;
            }
            instance.name = $"RedFoodMushroom_{redPositions.Count + 1:000}";
            instance.transform.position = candidate;
            FoodMushroom food = instance.GetComponent<FoodMushroom>();
            if (food != null)
            {
                food.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Red);
            }
            Undo.RegisterCreatedObjectUndo(instance, "Scatter red food mushroom");
            redPositions.Add(candidate);
        }

        if (redPositions.Count == 0)
        {
            Undo.DestroyObjectImmediate(root);
            Debug.LogError(
                "No valid red-mushroom positions were found inside the blue " +
                "population bounds. No scene objects were retained."
            );
            return;
        }

        Selection.activeGameObject = root;
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        string message =
            $"Repaired and expanded to {redPositions.Count}/{ScatterTargetCount} " +
            $"deterministic red mushrooms after {attempts} new-placement attempts. " +
            "Save the scene after visually inspecting terrain contact and colliders.";
        if (redPositions.Count < ScatterTargetCount)
        {
            Debug.LogWarning(message);
        }
        else
        {
            Debug.Log(message);
        }
    }

    private static bool IsSeparated(
        Vector3 candidate,
        List<Vector3> positions,
        float minimumDistance
    )
    {
        float minimumDistanceSquared = minimumDistance * minimumDistance;
        foreach (Vector3 position in positions)
        {
            float dx = candidate.x - position.x;
            float dz = candidate.z - position.z;
            if (dx * dx + dz * dz < minimumDistanceSquared)
            {
                return false;
            }
        }
        return true;
    }
}
