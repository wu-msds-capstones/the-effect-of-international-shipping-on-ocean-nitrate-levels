import csv
from osgeo import gdal
import os

gdal.UseExceptions()


def latlon_to_pixel(lat, lon, gt):
    # Convert the latitude and longitude to pixel coordinates
    x = (lon - gt[0]) / gt[1]
    y = (lat - gt[3]) / gt[5]
    return x, y

def get_pixel_value(x, y, ds):
    # Get the raster band
    band = ds.GetRasterBand(1)
    # Get the pixel value
    pixel_value = band.ReadAsArray(int(x), int(y), 1, 1)[0, 0]
    return pixel_value

def main(lat, lon, tiff_file):
    # Open the TIFF image
    ds = gdal.Open(tiff_file)
    # Get the geotransform
    gt = ds.GetGeoTransform()
    # Convert the latitude and longitude to pixel coordinates
    x, y = latlon_to_pixel(lat, lon, gt)
    # Get the pixel value
    pixel_value = get_pixel_value(x, y, ds)
    return pixel_value

tiff_file = 'shipdensity_global/shipdensity_global.tif'

if os.path.isfile('boat.csv') == False:
    with open('boat.csv', 'w',newline='\n') as coords:
        coords.write('Key,Density\n')
else:
    pass 


def csv_writer_boat(coords_file):
    with open(coords_file, 'r') as file:
        print('read successfully')
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    
    with open(coords_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'id':
                pass
            else:
                coord_id = int(row[0])
                latitude_value = float(row[1])
                longitude_value = float(row[2])
                if latitude_value >= 85 or latitude_value <= -85:
                    pv = ''
                else:
                    pv = main(lat=latitude_value, lon=longitude_value,tiff_file=tiff_file)
                newrow = list([coord_id,pv])
                with open('boat.csv', 'a', newline='\n') as bt:
                        writer = csv.writer(bt)
                        writer.writerow(newrow)
                current_row = reader.line_num
                print(f'{row_count} : {current_row}')

csv_writer_boat('coords.csv')

print('all done u rat')
