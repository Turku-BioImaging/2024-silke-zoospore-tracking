DROP TABLE IF EXISTS tracks;

DROP TABLE IF EXISTS particles;

CREATE TABLE
    tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replicate VARCHAR(255),
        sample VARCHAR(255),
        frame INT,
        particle INT,
        x REAL,
        y REAL,
        test VARCHAR(255),
        step_init INT,
        step_end INT,
        step_init_abs INT,
        step_end_abs INT,
        step_type VARCHAR(255),
        frame_interval REAL,
        dx_um REAL,
        dy_um REAL,
        displacement_um REAL,
        UNIQUE (replicate, sample, particle, frame)
    );

CREATE TABLE
    particles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replicate VARCHAR(255),
        sample VARCHAR(255),
        particle INT,
        average_speed REAL,
        curvilinear_velocity REAL,
        straight_line_velocity REAL,
        directionality_ratio REAL,
        UNIQUE (replicate, sample, particle)
    );

-- Create indexes for the tracks table
CREATE INDEX idx_tracks_replicate ON tracks (replicate);

CREATE INDEX idx_tracks_sample ON tracks (sample);

CREATE INDEX idx_tracks_particle ON tracks (particle);

CREATE INDEX idx_tracks_frame ON tracks (frame);

-- Create indexes for the particles table
CREATE INDEX idx_particles_replicate ON particles (replicate);

CREATE INDEX idx_particles_sample ON particles (sample);

CREATE INDEX idx_particles_particle ON particles (particle);