//
//  main.c
//  griddata
//
//  Created by Vivek Sinha on 31/07/2023.
//

#include <stdio.h>

int main(void)
{ float rf[135][129],rainfall;
float lo[135], la[129] ;
int i, j, k, year, year1, month, date, nd ;
FILE *fptr1,*fptr2;
int nd1[13] = {0,31,28,31,30,31,30,31,31,30,31,30,31} ;
int nd2[13] = {0,31,29,31,30,31,30,31,31,30,31,30,31} ;
year = 2016 ;
printf("Year = %d\n",year) ;
fptr1 = fopen("/Users/vsinha/Documents/dataset/Rainfall_ind2016_rfp25.grd","rb"); // input file
fptr2 = fopen("/Users/vsinha/Documents/dataset/Rain_2016.txt","w");
if(fptr1==NULL) { printf("Can't open file"); return 0; }
if(fptr2==NULL) { printf("Can't open file"); return 0; }
for(j=0 ; j < 135 ; j++) lo[j] = 66.5 + j * 0.25 ;
for(j=0 ; j < 129 ; j++) la[j] = 6.5 + j * 0.25 ;
//year1 = year / 4 ;
//year1 = year1 * 4 ;
for(month=1 ; month < 13 ; month++)
{ nd = nd2[month] ;
    for(date=1 ; date <= nd ; date++)
    {
//    { fprintf(fptr2,"\n%02d,%02d,%04d\t",date,month,year);
        for(j=0 ; j < 135 ; j++)
            //{ fprintf(fptr2,"%7.2f",lo[j]) ;
        {
            for(i=0 ; i < 129 ; i++)
                //{ fprintf(fptr2,"%8.2f",la[i]) ;
            {
                
                if(fread(&rainfall,sizeof(rainfall),1,fptr1) != 1) return 0 ;
                    rf[j][i] = rainfall ;
//                printf("%7.1f \n",rf[j][i]);
                if(rf[j][i] != -999.0)
                    if(la[i] > 20.10 && la[i] < 24.70 )
                        if(lo[j] > 68.5 && lo[j] < 74.5)
                    fprintf(fptr2,"%02d,%02d,%04d\t%7.2f,%8.2f,%7.1f\n",date,month,year,lo[j],la[i],rf[j][i] );
            
            }
            
        }
//        printf("%4d %02d %02d \n",year,month,date);
    }
    
}
fclose(fptr1);
fclose(fptr2);
printf("Year = %d",year) ;
return 0;
}
