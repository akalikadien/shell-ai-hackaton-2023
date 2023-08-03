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
#include <string.h>

int main(int argc, char* argv[])
{ float t;
float lo[31], la[31], T_temp=0.0 ;
int c = 1, i, j, month, date, nd ;
char* year ;
FILE *fptr1,*fptr2;
    printf("%s,%s,%s\n", argv[1],argv[2],argv[3]);
year = argv[1];
int nd1[13] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
    if(strcmp(argv[1],"2016") == 0 | strcmp(argv[1],"2012") == 0){
        nd1[2] = 29 ;
    }

printf("Year = %s\n",year) ;
//fptr1 = fopen("/Users/vsinha/Documents/dataset/Mintemp_MinT_2016.GRD","rb"); // input file
//fptr2 = fopen("/Users/vsinha/Documents/dataset/MinT_2016.csv","w");
fptr1 = fopen(argv[2],"rb");
fptr2 = fopen(argv[3], "w");
if(fptr1==NULL) { printf("Can't open file"); return 0; }
if(fptr2==NULL) { printf("Can't open file"); return 0; }
fprintf(fptr2,"%s,%s,%s,%s,%s\n","mm","yy","Longitude","Latitude","Temperature" );
for(j=0 ; j < 31 ; j++) lo[j] = 67.5 + j * 1.0 ;
for(j=0 ; j < 31 ; j++) la[j] = 7.5 + j * 1.0 ;

for(month=1 ; month < 13 ; month++)
{ nd = nd1[month] ;
    
    {
//    { fprintf(fptr2,"\n%02d,%02d,%04d\t",date,month,year);
        for(j=0 ; j < 31 ; j++)
            //{ fprintf(fptr2,"%7.2f",lo[j]) ;
        {
            for(i=0 ; i < 31 ; i++)
                //{ fprintf(fptr2,"%8.2f",la[i]) ;
            { T_temp = 0.0 ;
                c = 1;
                for(date=1 ; date <= nd ; date++)
                {
                    
                    if(fread(&t,sizeof(t),1,fptr1) != 1) return 0 ;
                    if(t < 90.00){
                        T_temp += t ;
                        c += 1;
                    }
                    
                }
                if(c != 1){
                    T_temp = T_temp / (c-1) ;
                }
                else {T_temp = T_temp / c ;}

//                printf("%7.1f \n",rf[j][i]);
                if(la[i] > 19.1 && la[i] < 25.7 && lo[j] > 67.5 && lo[j] < 75.5){
                    fprintf(fptr2,"%02d,%s,%7.2f,%8.2f,%7.1f\n",month,year,lo[j],la[i],T_temp );
                }
            
            }
            
        }
//        printf("%4d %02d %02d \n",year,month,date);
    }
    
}
fclose(fptr1);
fclose(fptr2);
printf("Year = %s",year) ;
return 0;
}
