//
//  main.c
//  temp
//
//  Created by Vivek Sinha on 31/07/2023.
//

//
//  main.c
//  griddata
//
//  Created by Vivek Sinha on 31/07/2023.
//

#include <stdio.h>

int main(void)
{ float rf[31][31],t;
float lo[31], la[31] ;
int i, j, year, month, date, nd ;
FILE *fptr1,*fptr2;
//int nd1[13] = {0,31,28,31,30,31,30,31,31,30,31,30,31} ;
int nd1[13] = {0,31,29,31,30,31,30,31,31,30,31,30,31} ;
year = 2010 ;
printf("Year = %d\n",year) ;
fptr1 = fopen("/Users/vsinha/Documents/dataset/Mintemp_MinT_2010.GRD","rb"); // input file
fptr2 = fopen("/Users/vsinha/Documents/dataset/MinT_2010.txt","w");
if(fptr1==NULL) { printf("Can't open file"); return 0; }
if(fptr2==NULL) { printf("Can't open file"); return 0; }
for(j=0 ; j < 31 ; j++) lo[j] = 67.5 + j * 1.0 ;
for(j=0 ; j < 31 ; j++) la[j] = 7.5 + j * 1.0 ;
//year1 = year / 4 ;
//year1 = year1 * 4 ;
for(month=1 ; month < 13 ; month++)
{ nd = nd1[month] ;
    for(date=1 ; date <= nd ; date++)
    {
//    { fprintf(fptr2,"\n%02d,%02d,%04d\t",date,month,year);
        for(j=0 ; j < 31 ; j++)
            //{ fprintf(fptr2,"%7.2f",lo[j]) ;
        {
            for(i=0 ; i < 31 ; i++)
                //{ fprintf(fptr2,"%8.2f",la[i]) ;
            {
                
                if(fread(&t,sizeof(t),1,fptr1) != 1) return 0 ;
                    rf[j][i] = t ;
//                printf("%7.1f \n",rf[j][i]);
                if(rf[j][i] < 99.90)
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
