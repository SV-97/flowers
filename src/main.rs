use std::ops::{Add, Rem};

use image::{imageops, GrayImage, Luma, Rgb, RgbImage};
use nalgebra::{DMatrix, DVector, Matrix3, Point2, RealField, Unit, Vector2, Vector4};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{num_traits::ToPrimitive, StandardNormal, Uniform};

pub trait Scalar: RealField + Copy + ToPrimitive {}
impl<T: RealField + Copy + ToPrimitive> Scalar for T {}

pub struct PerlinNoise<F> {
    /// grid of field vectors
    noise_grid: DMatrix<Unit<Vector2<F>>>,
    /// some interpolation function that's monotonic with h(0)=0, h(1)=1, h(1-x)=1-h(x)
    interpolation_func: fn(F) -> F,
}

pub fn deg5_smoothstep<F: Scalar>(x: F) -> F {
    let sq = x.powi(2);
    unsafe {
        let six = F::from_usize(6).unwrap_unchecked();
        let fifteen = F::from_usize(15).unwrap_unchecked();
        let ten = F::from_usize(10).unwrap_unchecked();
        sq * (six * sq.clone() - fifteen * x + ten)
    }
}

impl<F> PerlinNoise<F>
where
    F: Scalar,
    StandardNormal: Distribution<F>,
{
    pub fn new<R: Rng>(rng: &mut R, row_cells: usize, col_cells: usize) -> Self {
        let noise_grid_x: DMatrix<F> =
            DMatrix::from_distribution(row_cells + 1, col_cells + 1, &StandardNormal, rng);
        let noise_grid_y: DMatrix<F> =
            DMatrix::from_distribution(row_cells + 1, col_cells + 1, &StandardNormal, rng);
        // normalizing standard normal coords yields a uniform distribution over the sphere
        PerlinNoise {
            noise_grid: noise_grid_x.zip_map(&noise_grid_y, |x, y| {
                Unit::new_normalize(Vector2::new(x, y))
            }),
            interpolation_func: deg5_smoothstep,
        }
    }

    /// * `at` - pair of [y,x] coordinates
    pub fn sample(&self, at: Point2<F>, height: F, width: F) -> F {
        if at[0].is_negative() || at[0] >= height && at[1].is_negative() && at[1] >= width {
            panic!("Out of bounds")
        } else {
            // calculate the number of cells from the number of vertices in the given direction
            let (nrows, ncols) = self.noise_grid.shape();
            let row_cells = F::from_usize(nrows - 1).unwrap();
            let col_cells = F::from_usize(ncols - 1).unwrap();

            // determine the index of the cell in which the point is
            let vcell_idx = F::to_usize(&((row_cells * at[0]) / height).trunc()).unwrap();
            let hcell_idx = F::to_usize(&((col_cells * at[1]) / width).trunc()).unwrap();

            // get the "gradient vectors" corresponding to the vertics of the cell
            let grads = Vector4::new(
                self.noise_grid[(vcell_idx, hcell_idx)],
                self.noise_grid[(vcell_idx, hcell_idx + 1)],
                self.noise_grid[(vcell_idx + 1, hcell_idx)],
                self.noise_grid[(vcell_idx + 1, hcell_idx + 1)],
            );

            // get the "internal coordinates" of the vertices of the cell (vertices coords are 0,1)
            let verts: Vector4<Vector2<F>> = Vector4::new(
                Vector2::new(F::zero(), F::zero()),
                Vector2::new(F::zero(), F::one()),
                Vector2::new(F::one(), F::zero()),
                Vector2::new(F::one(), F::one()),
            );

            // compute the cell-internal coordinates of the point (same coordinatesystem as above)
            let internal_coord: Vector2<F> = Vector2::new(
                at[0].rem(height / row_cells) / (height / row_cells),
                at[1].rem(width / col_cells) / (width / col_cells),
            );

            // compute vectors from the vertices to the point (in internal coordinates)
            let to_point = Vector4::new(
                internal_coord - verts[0],
                internal_coord - verts[1],
                internal_coord - verts[2],
                internal_coord - verts[3],
            );

            // calculate dot products of those vectors with the gradients
            let s = Vector4::new(
                to_point[0].dot(&grads[0]),
                to_point[1].dot(&grads[1]),
                to_point[2].dot(&grads[2]),
                to_point[3].dot(&grads[3]),
            );

            // calculate interpolation values
            let h = [
                (self.interpolation_func)(internal_coord[0]),
                (self.interpolation_func)(internal_coord[1]),
            ];
            // calculate final value
            let f0 = s[0] * (F::one() - h[0]) + s[2] * h[0];
            let f1 = s[1] * (F::one() - h[0]) + s[3] * h[0];
            f0 * (F::one() - h[1]) + f1 * h[1]
        }
    }
}

fn float_to_luma8(x: f32) -> Luma<u8> {
    Luma([(255. * x.min(1.0)) as u8])
}

fn noise<R: RngCore>(
    width: usize,
    height: usize,
    mut rng: &mut R,
) -> (DMatrix<Vector2<f32>>, [GrayImage; 2]) {
    let nx = {
        [
            (0.5, PerlinNoise::new(&mut rng, 7, 5)),
            // (0.25, PerlinNoise::new(&mut rng, 6, 5)),
            (0.5 / 20., PerlinNoise::new(&mut rng, 20, 24)),
            (0.5 / 60., PerlinNoise::new(&mut rng, 75, 60)),
            // (0.5 / 100., PerlinNoise::new(&mut rng, 100, 120)),
        ]
    };
    let ny = {
        [
            (0.5, PerlinNoise::new(&mut rng, 15, 11)),
            (0.25, PerlinNoise::new(&mut rng, 5, 7)),
            // (0.5 / 20., PerlinNoise::new(&mut rng, 30, 21)),
            // (0.5 / 60., PerlinNoise::new(&mut rng, 40, 10)),
            (0.5 / 100., PerlinNoise::new(&mut rng, 110, 100)),
        ]
    };
    let m = DMatrix::from_fn(width, height, |x, y| {
        let p = Point2::new(y as f32, x as f32);

        Vector2::new(
            nx.iter()
                .map(|(w, n)| w * (1. + n.sample(p, height as f32, width as f32)) / 2.)
                .sum(),
            ny.iter()
                .map(|(w, n)| w * (1. + n.sample(p, height as f32, width as f32)) / 2.)
                .sum(),
        )
    });
    let imgs = [
        GrayImage::from_fn(width as u32, height as u32, |x, y| {
            float_to_luma8(m[(x as usize, y as usize)][0])
        }),
        GrayImage::from_fn(width as u32, height as u32, |x, y| {
            float_to_luma8(m[(x as usize, y as usize)][1])
        }),
    ];
    (m, imgs)
}

fn hsv_to_rgb(Hsv([hue, saturation, value]): Hsv) -> Rgb<u8> {
    let hue = hue * 360.;
    let c = value * saturation;
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = value - c;

    let (r, g, b) = if hue >= 0.0 && hue < 60.0 {
        (c, x, 0.0)
    } else if hue >= 60.0 && hue < 120.0 {
        (x, c, 0.0)
    } else if hue >= 120.0 && hue < 180.0 {
        (0.0, c, x)
    } else if hue >= 180.0 && hue < 240.0 {
        (0.0, x, c)
    } else if hue >= 240.0 && hue < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let r = ((r + m) * 255.0) as u8;
    let g = ((g + m) * 255.0) as u8;
    let b = ((b + m) * 255.0) as u8;

    Rgb([r, g, b])
}

// all values from 0 to 1
#[derive(Copy, Clone, Debug, PartialEq)]
struct Hsv([f32; 3]);

fn velocity_to_color(
    vel: f32,
    min_speed: f32,
    max_speed: f32,
    slow_color: &Hsv,
    fast_color: &Hsv,
) -> Rgb<u8> {
    // Normalize the velocity magnitude within the specified range
    let normalized_speed = (vel - min_speed) / (max_speed - min_speed);

    // Interpolate between slow_color and fast_color based on the normalized speed
    let hue = (1.0 - normalized_speed) * slow_color.0[0] + normalized_speed * fast_color.0[0];
    let saturation =
        (1.0 - normalized_speed) * slow_color.0[1] + normalized_speed * fast_color.0[1];
    let value = (1.0 - normalized_speed) * slow_color.0[2] + normalized_speed * fast_color.0[2];

    // Convert HSV to RGB using the hsv_to_rgb function
    hsv_to_rgb(Hsv([hue, saturation, value]))
}

/*
fn velocity_to_color(vel: f32, min_speed: f32, max_speed: f32) -> Rgb<u8> {
    // Define the range of hue values for blue to red
    let min_hue = 150. / 360.; // Blue
    let max_hue = 250. / 360.; // Red

    // Normalize the velocity magnitude within the specified range
    let normalized_speed = (vel - min_speed) / (max_speed - min_speed);

    // Interpolate between min_hue and max_hue based on the normalized speed
    let hue = min_hue + (max_hue - min_hue) * normalized_speed;

    // Set saturation and value to 1 for full saturation and brightness
    let saturation = 1.0;
    let value = 1.0;

    hsv_to_rgb(hue * 360.0, saturation, value)
}
*/

fn blend(start_color: Rgb<u8>, end_color: Rgb<u8>) -> Rgb<u8> {
    let t = 0.5; // Blend factor for one iteration (0.5 for an even blend)
    let r = (start_color.0[0] as f32 + (end_color.0[0] as f32 - start_color.0[0] as f32) * t) as u8;
    let g = (start_color.0[1] as f32 + (end_color.0[1] as f32 - start_color.0[1] as f32) * t) as u8;
    let b = (start_color.0[2] as f32 + (end_color.0[2] as f32 - start_color.0[2] as f32) * t) as u8;
    Rgb([r, g, b])
}

fn lerp(x: f32, from: [f32; 2], to: [f32; 2]) -> f32 {
    let m = (to[1] - to[0]) / (from[1] - from[0]);
    let result = m * (x - from[0]) + to[0];
    result
}

fn main() {
    let seed = thread_rng().gen();
    // #[rustfmt::skip]
    // let seed = [193, 52, 183, 122, 182, 126, 167, 189, 169, 78, 212, 231, 240, 44, 249, 63, 235, 83, 104, 174, 146, 37, 241, 143, 107, 196, 148, 74, 248, 143, 253, 100];
    println!("{:?}", seed);
    let mut rng = ChaCha8Rng::from_seed(seed);

    let width = 1024;
    let height = 1024;
    let (mut grads, [img_x, img_y]) = noise(width, height, &mut rng);
    img_x.save("nx.png").unwrap();
    img_y.save("ny.png").unwrap();
    let max = grads.map(|v| v.max()).max();
    let min = grads.map(|v| v.min()).min();
    grads = grads.map(|v| {
        Vector2::new(
            lerp(v[0], [min, max], [-1., 1.]),
            lerp(v[1], [min, max], [-1., 1.]),
        )
    });
    // dbg!(&grads);

    let img_width = 3840;
    let img_height = 2160;
    let background_color = Hsv([3.8 / 360., 1., 0.0]);
    let mut img = RgbImage::from_pixel(
        img_width as u32,
        img_height as u32,
        hsv_to_rgb(background_color),
    );

    let n_particles = 10_000;

    // generate random points in [0,1]²
    let unif_01 = Uniform::new_inclusive(0., 1.);
    let xs = DVector::from_distribution(n_particles, &unif_01, &mut rng);
    let ys = DVector::from_distribution(n_particles, &unif_01, &mut rng);
    let mut ppos: DVector<Vector2<f32>> = xs.zip_map(&ys, |x, y| Vector2::new(x, y));

    let unif_symm = Uniform::new_inclusive(-0.0005, 0.0005);
    let xs = DVector::from_distribution(n_particles, &unif_symm, &mut rng);
    let ys = DVector::from_distribution(n_particles, &unif_symm, &mut rng);
    let mut pvel: DVector<Vector2<f32>> = xs.zip_map(&ys, |x, y| Vector2::new(x, y));

    let eps_pos = 0.00025;
    let eps_vel = 0.1;
    let resistance = 0.03;

    let unif_symm = Uniform::new_inclusive(-0.5, 0.5);

    let slow_color = background_color;
    let fast_color = Hsv([28.8 / 360., 1., 1.]);

    for t in 0..1_000 {
        for (pos, vel) in ppos.iter_mut().zip(pvel.iter_mut()) {
            // find out indices into noise field for current particle positions
            let x_idx = (pos[0] * (width as f32 - 1.)) as usize;
            let y_idx = (pos[1] * (height as f32 - 1.)) as usize;
            // get corresponding gradients from field
            let vel_at_pos: Vector2<f32> = grads[(y_idx, x_idx)];
            // let phi = (t as f32) / 20_000. * f32::pi();
            // let vel_at_pos = Vector2::new(
            //     vel_at_pos[0] * f32::cos(phi) - vel_at_pos[1] * f32::sin(phi),
            //     vel_at_pos[0] * f32::sin(phi) + vel_at_pos[1] * f32::cos(phi),
            // );

            // let prev_vel_mag = vel.norm();
            // let random_dir = Vector2::from_distribution(&unif_symm, &mut rng);
            *vel = *vel + eps_vel * vel_at_pos; //.add(random_dir);
            *vel = *vel * (1. - resistance * vel.norm()).powi(2);
            *pos = (*pos + eps_pos * *vel).map(|x| x.rem(1.0));

            let img_x_idx = (pos[0] * (img_width as f32 - 1.)) as u32;
            let img_y_idx = (pos[1] * (img_height as f32 - 1.)) as u32;

            if t > 50 {
                let current = img.get_pixel_mut(img_x_idx, img_y_idx);
                let color = velocity_to_color(vel.norm(), 0., 1.0, &slow_color, &fast_color);
                *current = blend(*current, color);
            }
            /*
            let current = img.get_pixel_mut(img_x_idx, img_y_idx);
            current.0[0] = current.0[0].saturating_add((vel.norm() * 255.) as u8);
            */
            // let current = img.get_pixel_mut(img_x_idx, img_y_idx);
            // current.0[1] = current.0[1].saturating_add(30);
            // img.put_pixel(img_x_idx, img_y_idx, LumaA([128]));
        }
        // dbg!(&pvel);
    }

    #[rustfmt::skip]
    let id: Matrix3<f32> = Matrix3::new(
        0., 0., 0.,
        0., 1., 0.,
        0., 0., 0.,
    );

    #[rustfmt::skip]
    let gaussian: Matrix3<f32> = Matrix3::new(
        1./16., 1./8., 1./16.,
        1./8.,  1./4., 1./8.,
        1./16., 1./8., 1./16.,
    );

    let kernel: Matrix3<f32> = id + 0.5 * gaussian;

    imageops::filter3x3(&img, kernel.as_slice())
        .save("flow.png")
        .unwrap()
}
