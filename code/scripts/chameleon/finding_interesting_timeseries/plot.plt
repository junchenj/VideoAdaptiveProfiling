root="/mnt/tmp/diffs/"

set siz 20,3
set key bottom
#set title "Different resolutions"
set ylabel "Difference"
set xlabel "Time (minute)"
set xtics 10
set terminal x11 
set terminal postscript  eps color

time="18-01-09_08"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)




time="18-01-09_11"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)






time="18-01-09_17"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)






time="18-01-10_08"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)







time="18-01-10_11"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)










time="18-01-10_17"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)










time="18-01-11_08"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)










time="18-01-11_11"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)





time="18-01-11_17"
epsfile="time_series_".time."_Group_A.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.100-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.102-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.103-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "100" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "102" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "103" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_B.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.104-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.105-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "104" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "105" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_C.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.110-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.111-".time."*/log.txt")
filelist3=system("ls ".root."192.168.1.112-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "110" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "111" with lines lt 1 lc rgb 'blue' lw 3, \
for [filename in filelist3] filename u (($1)/25):2 title "112" with lines lt 1 lc rgb 'green' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)

epsfile="time_series_".time."_Group_D.eps"
set output "".epsfile
filelist1=system("ls ".root."192.168.1.115-".time."*/log.txt")
filelist2=system("ls ".root."192.168.1.116-".time."*/log.txt")
plot \
for [filename in filelist1] filename u (($1)/25):2 title "115" with lines lt 1 lc rgb 'red' lw 3, \
for [filename in filelist2] filename u (($1)/25):2 title "116" with lines lt 1 lc rgb 'blue' lw 3
set output
system('echo '.epsfile.'; epstopdf '.epsfile.'; rm '.epsfile)



