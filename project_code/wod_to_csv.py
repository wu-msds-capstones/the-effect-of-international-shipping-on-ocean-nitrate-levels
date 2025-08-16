
import os
import csv

if os.path.isfile('main.csv') == False:
    with open('main.csv', 'w') as dno:
        dno.write('cast,latitude,longitude,year,month,day,depth,depth_unit,depth_uncertainty,depth_uncertainty_unit,nitrate,nitrate_unit,measurement_type,plankton_value,plankton_unit,wod_pgc,itstsn\n')
else:
    pass

dat = ['1', '2']
dat.extend([str(i) for i in range(3,300)])

def csv_writer_nitrate(nitrate_file):
    with open(nitrate_file, 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)

    with open(nitrate_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0].strip() == 'CAST':
                current_cast = row[2].strip()
            if row[0].strip() == 'Latitude': 
                current_lat = str(round(float(row[2]), 2))
            if row[0].strip() == 'Longitude':
                current_lon = str(round(float(row[2]), 2))
            if row[0].strip() == 'Year':
                current_year = row[2].strip()
            if row[0].strip() == 'Month': 
                current_month = row[2].strip()
            if row[0].strip() == 'Day':
                current_day = row[2].strip()
            if row[0].strip() == 'UNITS':
                unit_d = row[1].strip()
                unit_u = row[2].strip()
                unit_n = row[5].strip()
            if row[0].strip() in dat: 
                current_depth = row[2].split()[0]
                current_un = row[3].split()[0]
                current_no3 = row[5].split()[0]
                newrow = list([current_cast, current_lat, current_lon, current_year, current_month, current_day, current_depth, unit_d, current_un, unit_u, current_no3, unit_n,'','','','',''])
                with open('main.csv', 'a') as dn:
                    writer = csv.writer(dn)
                    writer.writerow(newrow)
            current_row = reader.line_num
            print(f'{row_count} : {current_row}')


def csv_writer_plankton(plankton_file):
    with open(plankton_file, 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)

    with open(plankton_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            check = row[0].strip()
            if check == 'CAST':
                current_cast = row[2].strip()
            if check == 'Latitude': 
                current_lat = str(round(float(row[2]), 2))
            if check == 'Longitude':
                current_lon = str(round(float(row[2]), 2))
            if check == 'Year':
                current_year = row[2].strip()
            if check == 'Month': 
                current_month = row[2].strip()
            if check == 'Day':
                current_day = row[2].strip()
            if check in dat: 
                if row[3].split() != [] and row[4].split() != [] and row[11].split() != []:
                    current_measurement_type = row[3].split()[0]
                    current_value = row[4].split()[0]
                    current_unit = row[6].split()[0]
                    current_wodpgc = row[11].split()[0]
                    current_itstsn = row[12].split()[0]
                    newrow = list([current_cast, current_lat, current_lon, current_year, current_month, current_day, '', '', '', '', '', '',current_measurement_type,current_value,current_unit,current_wodpgc,current_itstsn])
                    with open('main.csv', 'a') as dn:
                        writer = csv.writer(dn)
                        writer.writerow(newrow)
            current_row = reader.line_num
            print(f'{row_count} : {current_row}')

nitrate_file1 = 'nitrate/wod_nitrate_1990_2025.OSD.csv'
nitrate_file2 = 'nitrate/wod_nitrate_1990_2025.PFL.csv'
nitrate_file3 = 'nitrate/wod_nitrate_1990_2025.CTD.csv'
plankton_file = 'plankton/wod_plankton.OSD.csv'

csv_writer_nitrate(nitrate_file=nitrate_file1)
csv_writer_nitrate(nitrate_file=nitrate_file2)
csv_writer_nitrate(nitrate_file=nitrate_file3)
csv_writer_plankton(plankton_file=plankton_file)

# test1 = 'nitrate/testfile.csv'
# test2 = 'plankton/testfile.csv'

# csv_writer_nitrate(test1)
# csv_writer_plankton(test2)

print('all done u rat')
