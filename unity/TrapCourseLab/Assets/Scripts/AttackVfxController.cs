using UnityEngine;

public class AttackVfxController : MonoBehaviour
{
    [SerializeField] private Transform punchAnchor;
    [SerializeField] private Transform kickAnchor;
    [SerializeField] private ParticleSystem punchEffect;
    [SerializeField] private ParticleSystem kickEffect;
    [SerializeField] private bool createDefaultEffects = true;
    [SerializeField] private float effectScale = 0.35f;

    private Vector3 punchLocalOffset;
    private Vector3 kickLocalOffset;
    private bool punchActive;
    private bool kickActive;
    private static Material sharedSoftParticleMaterial;
    private static Material sharedSoftTrailMaterial;

    private void Awake()
    {
        if (punchAnchor == null)
        {
            punchAnchor = FindAnchorFromHitbox("handhitbox", out punchLocalOffset);
        }

        if (kickAnchor == null)
        {
            kickAnchor = FindAnchorFromHitbox("leghitbox", out kickLocalOffset);
        }

        if (punchEffect == null && createDefaultEffects && punchAnchor != null)
        {
            punchEffect = CreateDefaultEffect("PunchPowerVFX", punchAnchor, punchLocalOffset, new Color(0.2f, 0.75f, 1f, 1f), false);
        }

        if (kickEffect == null && createDefaultEffects && kickAnchor != null)
        {
            kickEffect = CreateDefaultEffect("KickPowerVFX", kickAnchor, kickLocalOffset, new Color(0.9f, 0.25f, 1f, 1f), true);
        }

        SetPunchActive(false);
        SetKickActive(false);
    }

    public void SetPunchActive(bool active)
    {
        SetEffectActive(punchEffect, active, ref punchActive);
    }

    public void SetKickActive(bool active)
    {
        SetEffectActive(kickEffect, active, ref kickActive);
    }

    public void StopAll()
    {
        SetPunchActive(false);
        SetKickActive(false);
    }

    private void SetEffectActive(ParticleSystem effect, bool active, ref bool currentState)
    {
        if (effect == null || currentState == active)
        {
            return;
        }

        currentState = active;

        if (active)
        {
            effect.gameObject.SetActive(true);
            effect.Play(true);
            effect.Emit(35);
            return;
        }

        effect.Stop(true, ParticleSystemStopBehavior.StopEmitting);
    }

    private ParticleSystem CreateDefaultEffect(string effectName, Transform parent, Vector3 localOffset, Color color, bool denseKick)
    {
        GameObject effectObject = new GameObject(effectName);
        effectObject.transform.SetParent(parent, false);
        effectObject.transform.localPosition = localOffset;
        effectObject.transform.localRotation = Quaternion.identity;
        effectObject.transform.localScale = Vector3.one * effectScale;

        ParticleSystem particles = effectObject.AddComponent<ParticleSystem>();
        ParticleSystem.MainModule main = particles.main;
        main.loop = true;
        main.duration = 0.45f;
        main.startLifetime = denseKick ? new ParticleSystem.MinMaxCurve(0.22f, 0.48f) : new ParticleSystem.MinMaxCurve(0.18f, 0.42f);
        main.startSpeed = denseKick ? new ParticleSystem.MinMaxCurve(0.01f, 0.16f) : new ParticleSystem.MinMaxCurve(0.02f, 0.28f);
        main.startSize = denseKick ? new ParticleSystem.MinMaxCurve(0.26f, 0.54f) : new ParticleSystem.MinMaxCurve(0.22f, 0.48f);
        main.startColor = new ParticleSystem.MinMaxGradient(color, Color.white);
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.maxParticles = denseKick ? 360 : 260;

        ParticleSystem.EmissionModule emission = particles.emission;
        emission.rateOverTime = denseKick ? 260f : 180f;
        emission.SetBursts(new[]
        {
            denseKick ? new ParticleSystem.Burst(0f, 48, 70) : new ParticleSystem.Burst(0f, 28, 42),
        });

        ParticleSystem.ShapeModule shape = particles.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = denseKick ? 0.14f : 0.18f;

        ParticleSystem.VelocityOverLifetimeModule velocity = particles.velocityOverLifetime;
        velocity.enabled = true;
        velocity.space = ParticleSystemSimulationSpace.Local;
        velocity.radial = denseKick ? new ParticleSystem.MinMaxCurve(0.01f, 0.09f) : new ParticleSystem.MinMaxCurve(0.03f, 0.18f);
        velocity.orbitalX = new ParticleSystem.MinMaxCurve(0f, 0f);
        velocity.orbitalY = denseKick ? new ParticleSystem.MinMaxCurve(-1.05f, 1.05f) : new ParticleSystem.MinMaxCurve(-0.85f, 0.85f);
        velocity.orbitalZ = denseKick ? new ParticleSystem.MinMaxCurve(-0.85f, 0.85f) : new ParticleSystem.MinMaxCurve(-0.65f, 0.65f);

        ParticleSystem.ColorOverLifetimeModule colorOverLifetime = particles.colorOverLifetime;
        colorOverLifetime.enabled = true;
        Gradient gradient = new Gradient();
        gradient.SetKeys(
            new[]
            {
                new GradientColorKey(Color.white, 0f),
                new GradientColorKey(color, 0.25f),
                new GradientColorKey(color, 1f),
            },
            new[]
            {
                new GradientAlphaKey(0f, 0f),
                new GradientAlphaKey(1f, 0.08f),
                new GradientAlphaKey(0f, 1f),
            });
        colorOverLifetime.color = gradient;

        ParticleSystem.SizeOverLifetimeModule sizeOverLifetime = particles.sizeOverLifetime;
        sizeOverLifetime.enabled = true;
        AnimationCurve sizeCurve = new AnimationCurve(
            new Keyframe(0f, 0.55f),
            new Keyframe(0.12f, 1f),
            new Keyframe(0.72f, 0.8f),
            new Keyframe(1f, 0f));
        sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1f, sizeCurve);

        ParticleSystem.TrailModule trails = particles.trails;
        trails.enabled = true;
        trails.mode = ParticleSystemTrailMode.PerParticle;
        trails.ratio = 1f;
        trails.lifetime = 0.34f;
        trails.dieWithParticles = false;
        trails.widthOverTrail = new ParticleSystem.MinMaxCurve(0.34f, new AnimationCurve(
            new Keyframe(0f, 1f),
            new Keyframe(1f, 0f)));
        trails.colorOverTrail = new ParticleSystem.MinMaxGradient(color, new Color(1f, 1f, 1f, 0.65f));

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        renderer.renderMode = ParticleSystemRenderMode.Billboard;
        renderer.minParticleSize = 0.02f;
        renderer.maxParticleSize = 0.35f;
        renderer.material = GetSoftParticleMaterial();
        renderer.trailMaterial = GetSoftTrailMaterial();

        effectObject.SetActive(false);
        return particles;
    }

    private Transform FindAnchorFromHitbox(string searchText, out Vector3 localOffset)
    {
        localOffset = Vector3.zero;
        Transform[] children = GetComponentsInChildren<Transform>(true);
        for (int i = 0; i < children.Length; i++)
        {
            string childName = children[i].name.Replace(" ", string.Empty).ToLowerInvariant();
            if (childName.Contains(searchText))
            {
                localOffset = GetHitboxCenter(children[i]);

                if (children[i].gameObject.activeInHierarchy)
                {
                    return children[i];
                }

                if (children[i].parent != null)
                {
                    localOffset += children[i].localPosition;
                    return children[i].parent;
                }

                return children[i];
            }
        }

        return null;
    }

    private Vector3 GetHitboxCenter(Transform hitbox)
    {
        SphereCollider sphere = hitbox.GetComponent<SphereCollider>();
        if (sphere != null)
        {
            return Vector3.Scale(sphere.center, hitbox.localScale);
        }

        BoxCollider box = hitbox.GetComponent<BoxCollider>();
        if (box != null)
        {
            return Vector3.Scale(box.center, hitbox.localScale);
        }

        CapsuleCollider capsule = hitbox.GetComponent<CapsuleCollider>();
        if (capsule != null)
        {
            return Vector3.Scale(capsule.center, hitbox.localScale);
        }

        return Vector3.zero;
    }

    private static Material GetSoftParticleMaterial()
    {
        if (sharedSoftParticleMaterial != null)
        {
            return sharedSoftParticleMaterial;
        }

        sharedSoftParticleMaterial = CreateSoftMaterial("Generated Soft Energy Particle");
        return sharedSoftParticleMaterial;
    }

    private static Material GetSoftTrailMaterial()
    {
        if (sharedSoftTrailMaterial != null)
        {
            return sharedSoftTrailMaterial;
        }

        sharedSoftTrailMaterial = CreateSoftMaterial("Generated Soft Energy Trail");
        return sharedSoftTrailMaterial;
    }

    private static Material CreateSoftMaterial(string materialName)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
        if (shader == null)
        {
            shader = Shader.Find("Particles/Standard Unlit");
        }

        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }

        Material material = new Material(shader)
        {
            name = materialName,
            mainTexture = CreateGaussianTexture(64)
        };

        if (material.HasProperty("_BaseMap"))
        {
            material.SetTexture("_BaseMap", material.mainTexture);
        }

        if (material.HasProperty("_MainTex"))
        {
            material.SetTexture("_MainTex", material.mainTexture);
        }

        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", Color.white);
        }

        if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", Color.white);
        }

        ConfigureTransparentAdditiveMaterial(material);
        return material;
    }

    private static void ConfigureTransparentAdditiveMaterial(Material material)
    {
        if (material.HasProperty("_Surface"))
        {
            material.SetFloat("_Surface", 1f);
        }

        if (material.HasProperty("_Blend"))
        {
            material.SetFloat("_Blend", 2f);
        }

        if (material.HasProperty("_AlphaClip"))
        {
            material.SetFloat("_AlphaClip", 0f);
        }

        material.SetOverrideTag("RenderType", "Transparent");
        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.One);
        material.SetInt("_ZWrite", 0);
        material.DisableKeyword("_ALPHATEST_ON");
        material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }

    private static Texture2D CreateGaussianTexture(int size)
    {
        Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            name = "Generated Gaussian Particle Texture",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        Color[] pixels = new Color[size * size];
        float center = (size - 1) * 0.5f;
        float sigma = size * 0.18f;
        float twoSigmaSquared = 2f * sigma * sigma;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float dx = x - center;
                float dy = y - center;
                float distanceSquared = dx * dx + dy * dy;
                float alpha = Mathf.Exp(-distanceSquared / twoSigmaSquared);
                alpha = Mathf.Clamp01((alpha - 0.02f) / 0.98f);
                pixels[y * size + x] = new Color(1f, 1f, 1f, alpha);
            }
        }

        texture.SetPixels(pixels);
        texture.Apply(false, true);
        return texture;
    }
}
