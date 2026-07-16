using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class MinimalTrapCourseBuilder
{
    private const string ScenePath = "Assets/Scenes/TrapCourseLab.unity";
    private const string FoodPrefabPath = "Assets/TrapCourseFood.prefab";

    [MenuItem("Tiny Consciousness/Build Minimal Trap Course")]
    public static void Build()
    {
        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        Material robotMaterial = CreateMaterial("Robot Material", new Color(0.18f, 0.68f, 0.78f));
        Material foodMaterial = CreateMaterial("Food Material", new Color(0.40f, 1.00f, 0.48f));

        GameObject lightObject = new GameObject("Directional Light");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.15f;
        lightObject.transform.rotation = Quaternion.Euler(48f, -32f, 0f);

        GameObject robot = new GameObject("RAVEL Robot");
        robot.transform.position = new Vector3(0f, 0.25f, 0f);
        CharacterController capsule = robot.AddComponent<CharacterController>();
        capsule.radius = 0.34f;
        capsule.height = 1.7f;
        capsule.center = new Vector3(0f, 0.85f, 0f);
        ThirdPersonRobotController controller = robot.AddComponent<ThirdPersonRobotController>();
        RobotUdpBridge bridge = robot.AddComponent<RobotUdpBridge>();

        GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        body.name = "Robot Body";
        Object.DestroyImmediate(body.GetComponent<Collider>());
        body.transform.SetParent(robot.transform, false);
        body.transform.localPosition = new Vector3(0f, 0.85f, 0f);
        body.transform.localScale = new Vector3(0.55f, 0.85f, 0.55f);
        body.GetComponent<Renderer>().sharedMaterial = robotMaterial;

        GameObject cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.07f, 0.09f, 0.11f);
        cameraObject.AddComponent<AudioListener>();
        CameraLookAtCharacter cameraFollow = cameraObject.AddComponent<CameraLookAtCharacter>();
        SetObjectReference(cameraFollow, "target", robot.transform);
        SetObjectReference(controller, "cameraTransform", cameraObject.transform);
        SetEnum(controller, "controlMode", 1);

        FoodMushroom foodPrefab = CreateFoodPrefab(foodMaterial);

        GameObject courseObject = new GameObject("Trap Course Lab");
        TrapCourseSpawner spawner = courseObject.AddComponent<TrapCourseSpawner>();
        SetObjectReference(spawner, "robot", controller);
        SetObjectReference(spawner, "foodPrefab", foodPrefab);
        SetObjectReference(spawner, "courseCamera", cameraFollow);
        SetBool(spawner, "snapDeckToTerrain", false);
        SetBool(spawner, "buildOnStart", true);

        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log($"Minimal trap-course scene created at {ScenePath}");
    }

    public static void BuildSmokePlayer()
    {
        Build();
        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = "/tmp/TinyTrapCourseSmoke.app",
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.Development,
        };
        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
        {
            throw new System.InvalidOperationException($"Smoke player build failed: {report.summary.result}");
        }
        Debug.Log($"Smoke player build succeeded: {report.summary.totalSize} bytes");
    }

    private static FoodMushroom CreateFoodPrefab(Material material)
    {
        GameObject root = new GameObject("Trap Course Food");
        BoxCollider collider = root.AddComponent<BoxCollider>();
        collider.size = new Vector3(1.1f, 1.4f, 1.1f);
        collider.center = new Vector3(0f, 0.7f, 0f);
        root.AddComponent<Rigidbody>();
        root.AddComponent<FoodMushroom>();

        GameObject stem = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        Object.DestroyImmediate(stem.GetComponent<Collider>());
        stem.transform.SetParent(root.transform, false);
        stem.transform.localPosition = new Vector3(0f, 0.45f, 0f);
        stem.transform.localScale = new Vector3(0.22f, 0.45f, 0.22f);
        stem.GetComponent<Renderer>().sharedMaterial = material;

        GameObject cap = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Object.DestroyImmediate(cap.GetComponent<Collider>());
        cap.transform.SetParent(root.transform, false);
        cap.transform.localPosition = new Vector3(0f, 1.05f, 0f);
        cap.transform.localScale = new Vector3(1.05f, 0.42f, 1.05f);
        cap.GetComponent<Renderer>().sharedMaterial = material;

        GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, FoodPrefabPath);
        Object.DestroyImmediate(root);
        return prefab.GetComponent<FoodMushroom>();
    }

    private static Material CreateMaterial(string name, Color color)
    {
        string path = $"Assets/{name.Replace(' ', '_')}.mat";
        Material existing = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (existing != null)
        {
            return existing;
        }

        Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        Material material = new Material(shader) { color = color };
        AssetDatabase.CreateAsset(material, path);
        return material;
    }

    private static void SetObjectReference(Object target, string propertyName, Object value)
    {
        SerializedObject serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).objectReferenceValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void SetBool(Object target, string propertyName, bool value)
    {
        SerializedObject serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).boolValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void SetEnum(Object target, string propertyName, int value)
    {
        SerializedObject serialized = new SerializedObject(target);
        serialized.FindProperty(propertyName).enumValueIndex = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }
}
