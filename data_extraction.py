import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
import os
import h3
import shapely
import pyproj
import numpy as np
import json

import datetime

load_dotenv()

params = {
    'db': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'url': os.getenv('DATABASE_URL'),
    'sampling_mode': 'circle', # polygon | circle
    'h3_max_res': 7
}

QUERY = 'SELECT an."taxa_id", ah."hex_id" FROM "annotation" AS an INNER JOIN "annotation_hexagon" AS ah ON an."annotation_id"=ah."annotation_id"'

engine = sqlalchemy.engine.create_engine(url=params['url'])
df = pd.read_sql(QUERY, engine)

print('Loaded: ', len(df))

def calculate_geo_distance(loc1, loc2):
    lat1, lng1 = loc1
    lat2, lng2 = loc2

    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance = geod.inv(lons1=lng1, lats1=lat1, lons2=lng2, lats2=lat2)
    return distance

def generate_random_points_in_polygon(boundary, N):
    polygon = shapely.Polygon(boundary)
    min_x, min_y, max_x, max_y = polygon.bounds

    random_points = []
    while len(random_points) < N:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        point = shapely.Point(x, y)
        if polygon.contains(point):
            random_points.append((x, y))

    return random_points

def generate_random_points_in_circle(lat, lng, R, N):
    center_point = shapely.geometry.Point(lng, lat)
    random_points = []
    while len(random_points) < N:
        r = R * np.sqrt(np.random.uniform(0, 1)) # random distance from center
        theta = np.random.uniform(0, 2 * np.pi) # random degree
        
        x = lng + r * np.cos(theta) / (111320 * np.cos(lat * np.pi / 180))
        y = lat + r * np.sin(theta) / 111320
        
        point = shapely.geometry.Point(x, y)
        
        if point.distance(center_point) * 111320 <= R:
            random_points.append((y, x))
    
    return random_points

hex_resolution = [h3.h3_get_resolution(hex_id) for hex_id in df['hex_id']]
df['hex_resolution'] = hex_resolution

hex_boundary = [h3.h3_to_geo_boundary(hex_id, geo_json=False) for hex_id in df['hex_id']]
df['hex_boundary'] = hex_boundary

if params['sampling_mode'] == 'polygon':
    pass
else:
    center_point = [h3.h3_to_geo(hex_id) for hex_id in df['hex_id']]
    df['center_point'] = center_point

    radius = [min([calculate_geo_distance(r['center_point'], loc) for loc in r['hex_boundary']]) for _, r in df.iterrows()] # type: ignore
    df['R'] = radius

start_time = datetime.datetime.now()

psuedo_points = []
for i, r in df.iterrows():
    random_n_points = None
    N = params['h3_max_res'] - r['hex_resolution']
    if params['sampling_mode'] == 'polygon':
       random_n_points = generate_random_points_in_polygon(r['hex_boundary'], N)
    else:
        lat, lng = r['center_point']
        random_n_points = generate_random_points_in_circle(lat, lng, r['R'], N)

    for random_lat, random_lng in random_n_points:
        psuedo_point = {
            'taxon_id': r['taxa_id'],
            'latitude': random_lat,
            'longitude': random_lng
        }

        psuedo_points.append(psuedo_point)

df_psuedo_points = pd.DataFrame(psuedo_points)

end_time = datetime.datetime.now()
print('Executed in: ', (end_time - start_time))

print('Generated: ', len(df_psuedo_points))

with open("paths.json", 'r') as f:
    paths = json.load(f)

date_now = datetime.datetime.now()
df_psuedo_points.to_csv(os.path.join(paths['annotation'], str(date_now)+'.csv'))
