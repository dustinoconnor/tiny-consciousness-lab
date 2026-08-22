using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public class YellowFlowerPrefabTools : EditorWindow
{
    private const string SourcePath = "Assets/flower.glb";
    private const string PrefabPath = "Assets/YellowFoodFlower.prefab";
    private const string ScatterRootName = "Scattered Yellow Food Flowers";

    private Terrain terrain;
    private int flowerCount = 60;
    private int randomSeed = 20260814;
    private float minFlowerSpacing = 16f;
    private float minOtherFoodSpacing = 8f;
    private float slopeLimitDegrees = 30f;
    private float verticalOffset = 1.5f;

    [MenuItem("Tiny Consciousness/Food/Yellow Flower Placement Tool")]
    private static void Open()
    {
        GetWindow<YellowFlowerPrefabTools>("Yellow Flowers");
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Yellow Flower Population", EditorStyles.boldLabel);
        terrain = (Terrain)EditorGUILayout.ObjectField("Terrain", terrain, typeof(Terrain), true);
        flowerCount = EditorGUILayout.IntSlider("Flower count", flowerCount, 1, 300);
        randomSeed = EditorGUILayout.IntField("Random seed", randomSeed);
        minFlowerSpacing = EditorGUILayout.FloatField("Yellow spacing", minFlowerSpacing);
        minOtherFoodSpacing = EditorGUILayout.FloatField("Other-food spacing", minOtherFoodSpacing);
        slopeLimitDegrees = EditorGUILayout.Slider("Slope limit", slopeLimitDegrees, 0f, 60f);
        verticalOffset = EditorGUILayout.FloatField("Vertical offset", verticalOffset);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Create / Repair Prefab"))
            {
                CreateOrRepairPrefab();
            }
            if (GUILayout.Button("Scatter / Repair Population"))
            {
                ScatterOrRepairPopulation();
            }
        }

        if (GUILayout.Button("Clear Yellow Population"))
        {
            ClearYellowPopulation();
        }

        EditorGUILayout.HelpBox(
            "Creates a yellow-observable pickup prefab from flower.glb. " +
            "Scatter uses its own scene root, preserves the red and blue populations, " +
            "rejects steep terrain, and applies the adjustable Y offset.",
            MessageType.Info
        );
    }

    [MenuItem("Tiny Consciousness/Food/Create or Repair Yellow Flower Prefab")]
    private static void CreateOrRepairPrefab()
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("Exit Play mode before creating the yellow flower prefab.");
            return;
        }

        GameObject source = AssetDatabase.LoadAssetAtPath<GameObject>(SourcePath);
        if (source == null)
        {
            Debug.LogError($"Yellow flower source was not found at {SourcePath}.");
            return;
        }

        GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(source);
        if (instance == null)
        {
            Debug.LogError("Could not instantiate the imported yellow flower.");
            return;
        }

        try
        {
            PrefabUtility.UnpackPrefabInstance(
                instance,
                PrefabUnpackMode.Completely,
                InteractionMode.AutomatedAction
            );
            instance.name = "YellowFoodFlower";

            FoodMushroom food = instance.GetComponent<FoodMushroom>();
            if (food == null)
            {
                food = instance.AddComponent<FoodMushroom>();
            }
            food.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Yellow);

            if (instance.GetComponent<FlowerWindSway>() == null)
            {
                instance.AddComponent<FlowerWindSway>();
            }

            BoxCollider pickupCollider = instance.GetComponent<BoxCollider>();
            FitColliderToRenderers(instance, pickupCollider);
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
                $"Created {PrefabPath} with yellow observable metadata, a fitted " +
                "trigger collider, and a kinematic rigidbody. Inspect its scale and collider once."
            );
        }
        finally
        {
            Object.DestroyImmediate(instance);
        }
    }

    private static void FitColliderToRenderers(GameObject root, BoxCollider target)
    {
        Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
        {
            target.center = Vector3.zero;
            target.size = Vector3.one;
            return;
        }

        Vector3 localMin = new Vector3(
            float.PositiveInfinity,
            float.PositiveInfinity,
            float.PositiveInfinity
        );
        Vector3 localMax = new Vector3(
            float.NegativeInfinity,
            float.NegativeInfinity,
            float.NegativeInfinity
        );
        foreach (Renderer renderer in renderers)
        {
            Bounds bounds = renderer.bounds;
            Vector3 center = bounds.center;
            Vector3 extents = bounds.extents;
            for (int x = -1; x <= 1; x += 2)
            {
                for (int y = -1; y <= 1; y += 2)
                {
                    for (int z = -1; z <= 1; z += 2)
                    {
                        Vector3 worldCorner = center + Vector3.Scale(
                            extents,
                            new Vector3(x, y, z)
                        );
                        Vector3 localCorner = root.transform.InverseTransformPoint(worldCorner);
                        localMin = Vector3.Min(localMin, localCorner);
                        localMax = Vector3.Max(localMax, localCorner);
                    }
                }
            }
        }

        target.center = (localMin + localMax) * 0.5f;
        Vector3 fittedSize = localMax - localMin;
        target.size = new Vector3(
            Mathf.Max(fittedSize.x, 0.05f),
            Mathf.Max(fittedSize.y, 0.05f),
            Mathf.Max(fittedSize.z, 0.05f)
        );
    }

    private void ScatterOrRepairPopulation()
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("Exit Play mode before scattering yellow flowers.");
            return;
        }

        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        if (prefab == null)
        {
            Debug.LogError(
                $"Yellow flower prefab was not found at {PrefabPath}. " +
                "Click Create / Repair Prefab first."
            );
            return;
        }

        Terrain targetTerrain = terrain != null ? terrain : Terrain.activeTerrain;
        if (targetTerrain == null)
        {
            Debug.LogError("No active Terrain was found for yellow-flower scattering.");
            return;
        }

        FoodMushroom[] foods = Object.FindObjectsByType<FoodMushroom>(
            FindObjectsInactive.Exclude
        );
        List<Vector2> otherFoodPositions = new List<Vector2>();
        List<Vector2> bluePositions = new List<Vector2>();
        foreach (FoodMushroom food in foods)
        {
            if (food == null || IsUnderScatterRoot(food.transform))
            {
                continue;
            }
            Vector3 position = food.transform.position;
            otherFoodPositions.Add(new Vector2(position.x, position.z));
            if (food.ObservableFeature == "blue")
            {
                bluePositions.Add(new Vector2(position.x, position.z));
            }
        }

        if (!TryGetPopulationBounds(targetTerrain, bluePositions, out Rect bounds))
        {
            Debug.LogError(
                "At least two spatially distributed blue FoodMushrooms are required " +
                "to define the occupied terrain region."
            );
            return;
        }

        GameObject root = GameObject.Find(ScatterRootName);
        if (root == null)
        {
            root = new GameObject(ScatterRootName);
            Undo.RegisterCreatedObjectUndo(root, "Create yellow flower population");
        }

        List<Vector2> yellowPositions = new List<Vector2>();
        FoodMushroom[] existingYellow = root.GetComponentsInChildren<FoodMushroom>(true);
        foreach (FoodMushroom flower in existingYellow)
        {
            Vector3 repaired = flower.transform.position;
            repaired.y = targetTerrain.SampleHeight(repaired)
                + targetTerrain.transform.position.y
                + verticalOffset;
            Undo.RecordObject(flower.transform, "Repair yellow flower height");
            flower.transform.position = repaired;
            Undo.RecordObject(flower, "Configure yellow flower");
            flower.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Yellow);
            EditorUtility.SetDirty(flower);
            yellowPositions.Add(new Vector2(repaired.x, repaired.z));
        }

        TerrainData terrainData = targetTerrain.terrainData;
        Vector3 terrainOrigin = targetTerrain.transform.position;
        Vector3 terrainSize = terrainData.size;
        System.Random random = new System.Random(randomSeed);
        int attempts = 0;
        int maxAttempts = Mathf.Max(flowerCount * 200, 2000);
        while (yellowPositions.Count < flowerCount && attempts < maxAttempts)
        {
            attempts += 1;
            float worldX = Mathf.Lerp(bounds.xMin, bounds.xMax, (float)random.NextDouble());
            float worldZ = Mathf.Lerp(bounds.yMin, bounds.yMax, (float)random.NextDouble());
            Vector2 candidateXZ = new Vector2(worldX, worldZ);
            if (!HasClearance(candidateXZ, yellowPositions, minFlowerSpacing)
                || !HasClearance(candidateXZ, otherFoodPositions, minOtherFoodSpacing))
            {
                continue;
            }

            float normalizedX = Mathf.InverseLerp(
                terrainOrigin.x,
                terrainOrigin.x + terrainSize.x,
                worldX
            );
            float normalizedZ = Mathf.InverseLerp(
                terrainOrigin.z,
                terrainOrigin.z + terrainSize.z,
                worldZ
            );
            if (terrainData.GetSteepness(normalizedX, normalizedZ) > slopeLimitDegrees)
            {
                continue;
            }

            Vector3 candidate = new Vector3(worldX, terrainOrigin.y, worldZ);
            candidate.y = targetTerrain.SampleHeight(candidate) + terrainOrigin.y + verticalOffset;
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(
                prefab,
                root.transform
            );
            if (instance == null)
            {
                continue;
            }

            instance.name = $"YellowFoodFlower_{yellowPositions.Count + 1:000}";
            instance.transform.SetPositionAndRotation(
                candidate,
                Quaternion.Euler(0f, (float)random.NextDouble() * 360f, 0f)
            );
            FoodMushroom flower = instance.GetComponent<FoodMushroom>();
            if (flower != null)
            {
                flower.ConfigureObservableProfile(FoodMushroom.ObservableProfile.Yellow);
            }
            Undo.RegisterCreatedObjectUndo(instance, "Scatter yellow flower");
            yellowPositions.Add(candidateXZ);
        }

        Selection.activeGameObject = root;
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        string message =
            $"Repaired and expanded the yellow population to " +
            $"{yellowPositions.Count}/{flowerCount} after {attempts} placement attempts. " +
            "Save the scene after inspecting several flowers and their colliders.";
        if (yellowPositions.Count < flowerCount)
        {
            Debug.LogWarning(message);
        }
        else
        {
            Debug.Log(message);
        }
    }

    private static bool IsUnderScatterRoot(Transform item)
    {
        Transform current = item;
        while (current != null)
        {
            if (current.name == ScatterRootName)
            {
                return true;
            }
            current = current.parent;
        }
        return false;
    }

    private static bool TryGetPopulationBounds(
        Terrain targetTerrain,
        List<Vector2> positions,
        out Rect result
    )
    {
        result = default;
        if (positions.Count < 2)
        {
            return false;
        }

        float minX = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float minZ = float.PositiveInfinity;
        float maxZ = float.NegativeInfinity;
        foreach (Vector2 position in positions)
        {
            minX = Mathf.Min(minX, position.x);
            maxX = Mathf.Max(maxX, position.x);
            minZ = Mathf.Min(minZ, position.y);
            maxZ = Mathf.Max(maxZ, position.y);
        }

        Vector3 terrainOrigin = targetTerrain.transform.position;
        Vector3 terrainSize = targetTerrain.terrainData.size;
        minX = Mathf.Max(minX, terrainOrigin.x);
        maxX = Mathf.Min(maxX, terrainOrigin.x + terrainSize.x);
        minZ = Mathf.Max(minZ, terrainOrigin.z);
        maxZ = Mathf.Min(maxZ, terrainOrigin.z + terrainSize.z);
        if (maxX - minX < 1f || maxZ - minZ < 1f)
        {
            return false;
        }

        result = Rect.MinMaxRect(minX, minZ, maxX, maxZ);
        return true;
    }

    private static bool HasClearance(
        Vector2 candidate,
        List<Vector2> positions,
        float minimumDistance
    )
    {
        float minimumDistanceSquared = minimumDistance * minimumDistance;
        foreach (Vector2 position in positions)
        {
            if ((candidate - position).sqrMagnitude < minimumDistanceSquared)
            {
                return false;
            }
        }
        return true;
    }

    private static void ClearYellowPopulation()
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("Exit Play mode before clearing yellow flowers.");
            return;
        }

        GameObject existing = GameObject.Find(ScatterRootName);
        if (existing == null)
        {
            Debug.Log("No scattered yellow flower population was found.");
            return;
        }

        Undo.DestroyObjectImmediate(existing);
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log("Cleared the scattered yellow flower population.");
    }
}
