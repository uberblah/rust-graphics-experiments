extern crate glutin;
extern crate gl;
extern crate nalgebra as na;

use glutin::{WindowBuilder, Event, GlRequest, Api, VirtualKeyCode, ElementState};
use std::os::raw::*;
use std::ffi::CStr;
use std::ffi::CString;
use std::mem::size_of;
use std::time::Instant;
use na::{Matrix4, Vector3};
use na::geometry as geom;

const INIT_W: i32 = 600;
const INIT_H: i32 = 600;
const NPOINTS_SIDE: i32 = 150;
const POINT_SIZE: f32 = 5.0;

const INIT_FOV: f32 = 70.0;
const INIT_POS: (f32, f32, f32) = (0.0, 0.0, 5.0);
const LOOK_RATE: f32 = 1.0;
const MOVE_RATE: f32 = 0.5;

fn main() {
    let window = WindowBuilder::new()
        .with_dimensions(INIT_W as u32, INIT_H as u32)
        .with_title("Glutin Window")
        .with_gl(GlRequest::Specific(Api::OpenGl, (3, 3)))
        .build().unwrap();

    unsafe {
        window.make_current().unwrap();
    }

    println!("configuring OpenGL");
    gl::load_with(|s: &str| window.get_proc_address(s) as *const c_void);

    let gl_string = |x: *const u8| unsafe {
        CStr::from_ptr(x as *const c_char).to_str().unwrap()
    };

    let vertex_source = CString::new("#version 330
layout(location = 0) in vec2 vp;
uniform float time;
uniform mat4 view;
uniform mat4 projection;
out vec3 offset;
out vec2 origPos;

void main() {
  origPos = vp;
  offset = noise3(vec3(vp * 2.0, time * 0.2));
  mat4 mvp = /*model * */ projection * view;
  gl_Position = mvp * vec4(vec3(vp, 0.0) + (offset * 0.3), 1.0);
}
").unwrap();
    let fragment_source = CString::new("#version 330
in vec3 offset;
in vec2 origPos;
out vec4 frag_color;

void main() {
  frag_color = vec4((offset + 1.0) / 2.0, 1.0);
}
").unwrap();

    let shader_program: u32;

    let mut vbo: u32 = 0;
    let mut vao: u32 = 0;
    let uni_time;
    let uni_projmat;
    let uni_viewmat;

    let mut tri_data: Vec<f32> = Vec::new();
    for x in -NPOINTS_SIDE..NPOINTS_SIDE {
        let xa = (x as f32) / NPOINTS_SIDE as f32;
        for y in -NPOINTS_SIDE..NPOINTS_SIDE {
            let ya = (y as f32) / NPOINTS_SIDE as f32;
            tri_data.push(xa);
            tri_data.push(ya);
        }
    }

    unsafe {
        println!("renderer: {}", gl_string(gl::GetString(gl::RENDERER)));
        println!("version: {}", gl_string(gl::GetString(gl::VERSION)));

        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            tri_data.len() as isize * size_of::<f32>() as isize,
            (tri_data.as_slice()).as_ptr() as *const c_void,
            gl::STATIC_DRAW
        );

        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);
        gl::EnableVertexAttribArray(0);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, 0 as *const c_void);

        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(vertex_shader, 1, &(vertex_source.as_ptr() as *const c_char), 0 as *const i32);
        gl::ShaderSource(fragment_shader, 1, &(fragment_source.as_ptr() as *const c_char), 0 as *const i32);
        gl::CompileShader(vertex_shader);
        let mut success: i32 = 0;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success == gl::FALSE as i32 {
            let mut log_len = 0;
            gl::GetShaderiv(vertex_shader, gl::INFO_LOG_LENGTH, &mut log_len);
            let mut copied_len = 0;
            let mut infolog = vec![0 as u8; log_len as usize];
            gl::GetShaderInfoLog(
                vertex_shader, log_len, &mut copied_len,
                infolog.as_mut_slice().as_mut_ptr() as *mut c_char
            );
            panic!("ERROR COMPILING VERTEX SHADER\n{}", String::from_utf8_unchecked(infolog));
        }

        gl::CompileShader(fragment_shader);
        let mut success: i32 = 0;
        gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
        if success == gl::FALSE as i32 {
            let mut log_len = 0;
            gl::GetShaderiv(fragment_shader, gl::INFO_LOG_LENGTH, &mut log_len);
            let mut copied_len = 0;
            let mut infolog = vec![0 as u8; log_len as usize];
            gl::GetShaderInfoLog(
                fragment_shader, log_len, &mut copied_len,
                infolog.as_mut_slice().as_mut_ptr() as *mut c_char
            );
            panic!("ERROR COMPILING FRAGMENT SHADER\n{}", String::from_utf8_unchecked(infolog));
        }

        shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vertex_shader);
        gl::AttachShader(shader_program, fragment_shader);
        gl::LinkProgram(shader_program);
        let mut success: i32 = 0;
        gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
        if success == gl::FALSE as i32 {
            let mut log_len = 0;
            gl::GetProgramiv(shader_program, gl::INFO_LOG_LENGTH, &mut log_len);
            let mut copied_len = 0;
            let mut infolog = vec![0 as u8; log_len as usize];
            gl::GetProgramInfoLog(
                shader_program, log_len, &mut copied_len,
                infolog.as_mut_slice().as_mut_ptr() as *mut c_char
            );
            panic!("ERROR LINKING SHADER PROGRAM\n{}", String::from_utf8_unchecked(infolog));
        }

        gl::UseProgram(shader_program);
        uni_time = gl::GetUniformLocation(shader_program, CString::new("time").unwrap().as_ptr());
        uni_viewmat = gl::GetUniformLocation(shader_program, CString::new("view").unwrap().as_ptr());
        uni_projmat = gl::GetUniformLocation(shader_program, CString::new("projection").unwrap().as_ptr());
        gl::UniformMatrix4fv(uni_viewmat, 1, false as u8, Matrix4::identity().as_slice().as_ptr());
        gl::UniformMatrix4fv(uni_projmat, 1, false as u8, Matrix4::identity().as_slice().as_ptr());

        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);
        gl::PointSize(POINT_SIZE);
    }

    let timer = Instant::now();
    let mut last_frame: f64 = 0.0;

    let mut fov = INIT_FOV;
    let mut view_matrix: Matrix4<f32> = Matrix4::new_translation(&Vector3::new(-INIT_POS.0, -INIT_POS.1, -INIT_POS.2));;
    let mut projection_matrix: Matrix4<f32> = Matrix4::identity();

    let (mut w, mut h) = (INIT_W, INIT_H);
    let mut resize = |new_w: i32, new_h: i32| {
        unsafe {
            w = new_w;
            h = new_h;
            gl::Viewport(0, 0, w, h);
            projection_matrix = Matrix4::new_perspective(
                w as f32 / h as f32, fov, 0.1, 100.0
            );
            // projection_matrix = Matrix4::new_orthographic(
            //     -1.0, 1.0, -1.0, 1.0, -100.0, 100.0
            // );
            gl::UniformMatrix4fv(uni_projmat, 1, false as u8, projection_matrix.as_slice().as_ptr());
        }
    };

    if let Some((initw, inith)) = window.get_inner_size_pixels() {
        resize(initw as i32, inith as i32);
    } else {
        panic!("the window was destroyed before we could use it");
    }

    let (mut move_l, mut move_r, mut move_f, mut move_b, mut move_u, mut move_d) =
        (false, false, false, false, false, false);
    let (mut turn_l, mut turn_r, mut look_u, mut look_d) =
        (false, false, false, false);

    'mainloop: loop {

        let elapsed = timer.elapsed();
        let time = elapsed.as_secs() as f64 + (elapsed.subsec_nanos() as f64 / 1000000000.0);
        let dt = (time - last_frame) as f32;

        for event in window.poll_events() {
            match event {
                Event::Closed => {break 'mainloop}
                Event::Resized(new_w, new_h) => {
                    resize(new_w as i32, new_h as i32);
                }
                Event::KeyboardInput(state, key, keycode) => {
                    match keycode {
                        Some(code) => {
                            println!("key {:?} {:?}", code, state);
                            let pressed = state == ElementState::Pressed;
                            match code {
                                VirtualKeyCode::Escape => {break 'mainloop;}

                                VirtualKeyCode::W        => {move_f = pressed;}
                                VirtualKeyCode::S        => {move_b = pressed;}
                                VirtualKeyCode::A        => {move_l = pressed;}
                                VirtualKeyCode::D        => {move_r = pressed;}
                                VirtualKeyCode::LShift   => {move_u = pressed;}
                                VirtualKeyCode::LControl => {move_d = pressed;}

                                VirtualKeyCode::Up    => {look_d = pressed;}
                                VirtualKeyCode::Down  => {look_u = pressed;}
                                VirtualKeyCode::Right => {turn_r = pressed;}
                                VirtualKeyCode::Left  => {turn_l = pressed;}
                                _ => ()
                            }
                        }
                        None => {
                            println!("unknown key {:?} {:?}", key, state);
                        }
                    }
                }
                _ => ()
            }
        }

        let mut rel_move = (0.0, 0.0, 0.0);
        let cur_move = MOVE_RATE * dt;
        if move_r {rel_move.0 += cur_move;}
        if move_l {rel_move.0 -= cur_move;}
        if move_u {rel_move.1 += cur_move;}
        if move_d {rel_move.1 -= cur_move;}
        if move_b {rel_move.2 += cur_move;}
        if move_f {rel_move.2 -= cur_move;}
        let mat_move = Matrix4::new_translation(&Vector3::new(-rel_move.0, -rel_move.1, -rel_move.2));

        let mut add_rot = (0.0, 0.0);
        let cur_rot = LOOK_RATE * dt;
        if look_u {add_rot.0 -= cur_rot;}
        if look_d {add_rot.0 += cur_rot;}
        if turn_r {add_rot.1 -= cur_rot;}
        if turn_l {add_rot.1 += cur_rot;}
        let mat_xrot = Matrix4::new_rotation(Vector3::new(-add_rot.0, 0.0, 0.0));
        let mat_yrot = Matrix4::new_rotation(Vector3::new(0.0, -add_rot.1, 0.0));
        let mat_rot = &mat_xrot * &mat_yrot;

        view_matrix = &mat_move * &mat_rot * &view_matrix;

        unsafe {
            gl::Uniform1f(uni_time, time as f32);
            gl::UniformMatrix4fv(uni_viewmat, 1, false as u8, view_matrix.as_slice().as_ptr());

            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            gl::DrawArrays(gl::POINTS, 0, (tri_data.len() / 2) as i32);
        }

        window.swap_buffers().unwrap();
        last_frame = time;
    }
}
