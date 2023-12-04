Use SalDB

-----------------Serotypes
select Interval, Montevideo, Dublin, Typhimurium, Total, 
       Montevideo*100.0/Total as MontevideoPRC,
       Dublin*100.0/Total as DublinPRC,
       Typhimurium*100.0/Total as TyphimuriumPRC
from 
 (select Interval,
       sum(case when [SEROTYPE] = 'Montevideo'  then 1 else 0 end) as Montevideo,
       sum(case when [SEROTYPE] = 'Dublin'      then 1 else 0 end) as Dublin,
       sum(case when [SEROTYPE] = 'Typhimurium' then 1 else 0 end) as Typhimurium,
       count(*) as Total
  from (select distinct id,
               case when serotype in ('Montevideo','Dublin','Typhimurium') then serotype
	            else 'Other'
	       end as serotype,
	       case when [Year] between 2010 and 2011 then '2010_2011'
	            when [Year] between 2012 and 2013 then '2012_2013'
	       end as Interval 
        From [NARM_Cattle_S_HACCP_2010_2013_View]) a
  group by Interval ) res
