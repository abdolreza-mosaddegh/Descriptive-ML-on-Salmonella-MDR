Use SalDB

-----------------Prevalence  
  select SEROTYPE, Year,
       (sum(resistant)*100.0) / (sum(resistant) + sum(susceptible)) as Prevalence
from
(SELECT      case when Year< 2012 then 'Bef2012' else 'Aft2012'  end as year,    case when SEROTYPE = 'dublin' then SEROTYPE else 'Non-Dublin' end as SEROTYPE
, case when status = 'susceptible' then 1 else 0 end as susceptible
, case when status = 'susceptible' then 0 else 1 end as resistant
  FROM NARM_Cattle_S_HACCP_2010_2013_View 
   ) a

   group by  SEROTYPE,Year 
 
    order by   SEROTYPE ,Year
