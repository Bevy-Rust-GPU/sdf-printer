use std::ops::{Add, Div, Mul, Sub};

use rust_gpu_bridge::glam::{Mat4, UVec2, Vec2, Vec3};
use rust_gpu_sdf::prelude::{
    default, squircle::Squircle, superellipse::Superellipse, superellipsoid::Superellipsoid,
    AttrDistance, Capsule, ChebyshevMetric, Circle, Cube, Decagon, Extrude, Field as FieldFunction,
    Hexagon, Line, Nonagon, Octagon, Octahedron, Pentagon, Plane, Point, Position, Quadrilateral,
    RaycastInput, Rotate3d, Septagon, Sphere, SphereTraceLipschitz, Square, Sweep, TaxicabMetric,
    Torus, Translate, Triangle, D2, D3, RaycastOutput, Raycast,
};
use type_fields::field::Field;

const ASCII_RAMP: &[&str] = &[" ", ".", ":", "-", "=", "+", "*", "#", "%", "@", "█"];

pub trait RemapRange:
    Sized
    + Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
    fn remap_range(self, from1: Self, to1: Self, from2: Self, to2: Self) -> Self {
        (self - from1) / (to1 - from1) * (to2 - from2) + from2
    }
}

impl<T> RemapRange for T where
    T: Copy + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>
{
}

pub trait TriangleWave {
    fn triangle_wave(self) -> Self;
}

impl TriangleWave for f32 {
    fn triangle_wave(self) -> Self {
        1.0 - 2.0 * ((self / 2.0).round() - (self / 2.0)).abs()
    }
}

pub trait SdfDimension {
    fn from_vec2(v: Vec2) -> Self;
}

impl SdfDimension for Vec2 {
    fn from_vec2(v: Vec2) -> Self {
        v
    }
}

impl SdfDimension for Vec3 {
    fn from_vec2(v: Vec2) -> Self {
        v.extend(0.0)
    }
}

pub trait AsciiRasterizer<Dim> {
    fn plot<Sdf>(&self, sdf: Sdf, x: i32, y: i32) -> f32
    where
        Sdf: FieldFunction<AttrDistance<Dim>>;

    fn rasterize<Sdf>(&self, sdf: Sdf, extent: UVec2) -> String
    where
        Sdf: Clone + FieldFunction<AttrDistance<Dim>>,
    {
        let mut out = String::default();

        let extent = extent.as_ivec2();
        for y in -extent.y..=extent.y {
            let mut line = String::default();
            for x in -extent.x..=extent.x {
                let dist = self.plot(sdf.clone(), x, y);

                let idx = (dist
                    .triangle_wave()
                    .remap_range(0.0, 1.0, 0.0, (ASCII_RAMP.len() - 1) as f32)
                    .round() as usize)
                    .clamp(0, ASCII_RAMP.len() - 1);

                let c = ASCII_RAMP[idx];
                let ansi = if dist >= 0.0 {
                    if idx == ASCII_RAMP.len() - 1 && dist <= 1.0 {
                        c.to_string()
                    } else {
                        ansi_term::Style::new()
                            .fg(ansi_term::Color::Blue)
                            .paint(c)
                            .to_string()
                    }
                } else if idx == ASCII_RAMP.len() - 1 && dist >= -1.0 {
                    c.to_string()
                } else {
                    ansi_term::Style::new()
                        .fg(ansi_term::Color::White)
                        .paint(c)
                        .to_string()
                };
                line += &ansi;
            }
            line += "\n";
            out += line.as_str();
        }

        out
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SdfSampler {
    pub view: Mat4,
    pub scale: Vec2,
}

impl Default for SdfSampler {
    fn default() -> Self {
        SdfSampler {
            view: Mat4::default(),
            scale: Vec2::ONE,
        }
    }
}

impl AsciiRasterizer<Vec2> for SdfSampler {
    fn plot<Sdf>(&self, sdf: Sdf, x: i32, y: i32) -> f32
    where
        Sdf: FieldFunction<AttrDistance<Vec2>>,
    {
        *sdf.field(&Position(Vec2::new(
            x as f32 * self.scale.x,
            y as f32 * self.scale.y,
        )))
    }
}

impl AsciiRasterizer<Vec3> for SdfSampler {
    fn plot<Sdf>(&self, sdf: Sdf, x: i32, y: i32) -> f32
    where
        Sdf: FieldFunction<AttrDistance<Vec3>>,
    {
        *sdf.field(&Position(self.view.transform_point3(Vec3::new(
            x as f32 * self.scale.x,
            y as f32 * self.scale.y,
            0.0,
        ))))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SdfRaymarcher<Raymarcher> {
    raymarcher: Raymarcher,
    view: Mat4,
    proj: Mat4,
}

impl<Raymarcher> Default for SdfRaymarcher<Raymarcher>
where
    Raymarcher: Default,
{
    fn default() -> Self {
        SdfRaymarcher {
            raymarcher: default(),
            view: Mat4::look_at_lh(Vec3::new(1.0, 1.0, 2.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_infinite_lh(
                core::f32::consts::FRAC_PI_2,
                2.0 * (14.0 / 32.0),
                0.05,
            ),
        }
    }
}

impl<Raymarcher> AsciiRasterizer<Vec3> for SdfRaymarcher<Raymarcher> {
    fn plot<Sdf>(&self, sdf: Sdf, x: i32, y: i32) -> f32
    where
        Sdf: FieldFunction<AttrDistance<Vec3>>,
    {
        let near = 0.05;
        let far = 1000.0;

        let eye = self.view.inverse().transform_point3(Vec3::ZERO);

        let dir = Vec3::new(x as f32 / 32.0, y as f32 / 14.0, -1.0);
        let dir = self.proj.project_point3(dir);
        let dir = self.view.transpose().transform_vector3(dir.normalize());

        let input = RaycastInput {
            start: near,
            end: far,
            eye,
            dir,
        };
        let out: RaycastOutput = FieldFunction::<Raycast>::field(&SphereTraceLipschitz::<150, _> {
            target: sdf,
            op: default(),
        }, &input);

        (1.0 - (out.steps as f32 / 150.0)) * if out.hit() { -1.0 } else { 1.0 }
    }
}

fn main() {
    let viewport = UVec2::new(32, 14);
    let aspect = Vec2::new(1.0, 2.0);
    let scale = Vec2::splat(1.0 / viewport.y as f32) * aspect;

    fn print_sdf<Dim, Sdf, Rast>(title: &str, sdf: Sdf, rasterizer: Rast, extent: UVec2)
    where
        Sdf: Clone + FieldFunction<AttrDistance<Dim>>,
        Rast: AsciiRasterizer<Dim>,
    {
        println!("{title:}");
        println!("{}", rasterizer.rasterize(sdf, extent));
    }

    let sampler = SdfSampler {
        view: Mat4::look_to_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0).normalize(), Vec3::Y),
        scale,
        ..default()
    };
    //let raymarcher = SdfRaymarcher::<SphereTraceLipschitz<150, _>>::default();

    println!("2D");

    println!("Bounds");
    print_sdf::<D2, _, _>(
        "Taxicab Metric",
        TaxicabMetric::default(),
        sampler,
        viewport,
    );
    print_sdf::<D2, _, _>(
        "Chebyshev Metric",
        ChebyshevMetric::default(),
        sampler,
        viewport,
    );

    println!("Fields");
    print_sdf::<D2, _, _>("Point", Point::default(), sampler, viewport);
    print_sdf("Line", Line::<D2>::default(), sampler, viewport);
    print_sdf("Plane", Plane::<D2>::default(), sampler, viewport);
    print_sdf::<D2, _, _>("Circle", Circle::default(), sampler, viewport);
    print_sdf("Capsule", Capsule::<D2>::default(), sampler, viewport);
    print_sdf("Square", Square::default(), sampler, viewport);
    print_sdf("Triangle", Triangle::triangle(), sampler, viewport);
    print_sdf(
        "Quadrilateral",
        Quadrilateral::quadrilateral(),
        sampler,
        viewport,
    );
    print_sdf("Pentagon", Pentagon::pentagon(), sampler, viewport);
    print_sdf("Hexagon", Hexagon::hexagon(), sampler, viewport);
    print_sdf("Septagon", Septagon::septagon(), sampler, viewport);
    print_sdf("Octagon", Octagon::octagon(), sampler, viewport);
    print_sdf("Nonagon", Nonagon::nonagon(), sampler, viewport);
    print_sdf("Decagon", Decagon::decagon(), sampler, viewport);

    print_sdf("Squircle", Squircle::default(), sampler, viewport);

    print_sdf(
        "Superellipse",
        Superellipse::default().with(Superellipse::n, 2.0 / 0.75),
        sampler,
        viewport,
    );

    println!("3D Raymarched");
    let raymarcher = sampler;

    println!("Bounds");
    print_sdf::<D3, _, _>(
        "Taxicab Metric",
        TaxicabMetric::default(),
        raymarcher,
        viewport,
    );
    print_sdf::<D3, _, _>(
        "Chebyshev Metric",
        ChebyshevMetric::default(),
        raymarcher,
        viewport,
    );

    println!("Fields");
    print_sdf::<D3, _, _>("Point", Point::default(), raymarcher, viewport);
    print_sdf("Line", Line::<D3>::default(), raymarcher, viewport);
    print_sdf("Plane", Plane::<D3>::default(), raymarcher, viewport);
    print_sdf::<D3, _, _>("Sphere", Sphere::default(), raymarcher, viewport);
    print_sdf("Capsule", Capsule::<D3>::default(), raymarcher, viewport);
    print_sdf("Cube", Cube::default(), raymarcher, viewport);
    print_sdf("Octahedron", Octahedron::default(), raymarcher, viewport);

    print_sdf::<D3, _, _>(
        "Torus",
        Torus::default()
            .with((Torus::core, Circle::radius), 0.75)
            .with((Torus::shell, Circle::radius), 0.25),
        raymarcher,
        viewport,
    );

    print_sdf::<D3, _, _>(
        "Extruded Circle",
        Extrude::<Circle>::default().with(Extrude::depth, 0.5),
        raymarcher,
        viewport,
    );

    Rotate3d::<Sweep<Point, Translate<_, Square>>>::default()
        .with(
            (Rotate3d::target, Sweep::shell, Translate::translation),
            Vec2::X,
        )
        .with(
            (
                Rotate3d::target,
                Sweep::shell,
                Translate::target,
                Square::extent,
            ),
            Vec2::ONE * 0.25,
        );

    print_sdf(
        "Revolved Offset Square",
        Rotate3d::<Sweep<Point, Translate<_, Square>>>::default()
            .with(
                (Rotate3d::target, Sweep::shell, Translate::translation),
                Vec2::X,
            )
            .with(
                (
                    Rotate3d::target,
                    Sweep::shell,
                    Translate::target,
                    Square::extent,
                ),
                Vec2::ONE * 0.25,
            ),
        raymarcher,
        viewport,
    );

    print_sdf(
        "Compositional Torus",
        Sweep::<Square, Circle>::default().with((Sweep::shell, Circle::radius), 0.25),
        raymarcher,
        viewport,
    );

    print_sdf(
        "Superellipsoid",
        Superellipsoid::default()
            .with(Superellipsoid::e1, 2.0 / 0.75)
            .with(Superellipsoid::e2, 2.0 / 0.75),
        raymarcher,
        viewport,
    );
}
