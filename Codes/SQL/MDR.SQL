Use SalDB

-----------------MDR
select MDR_by_decade.SEROTYPE, MDR_by_decade.year,  
       MultiResistantIsolates,TotalIsolates, (MultiResistantIsolates * 100.0) /TotalIsolates as MultiDrugResistant_Ratio
from

(select SEROTYPE, year,   count(*) as MultiResistantIsolates 
from
(select SEROTYPE,year, id
 from
 (SELECT distinct id ,case when Year<2012 then 'Before2012' else 'After2012' end as year, case when SEROTYPE = 'dublin' then SEROTYPE else 'Non-Dublin' end as SEROTYPE
      ,  case when   class in    ('Cephalosporins','Penicillins', 'Beta lactam','Carbapenems' ) then 'Beta lactam' else class end  as class
  FROM NARM_Cattle_S_HACCP_2010_2013_View 
  where Status = 'resistant'  ) ResistantIsloates
  group by   SEROTYPE, year,id
  having count(*) > 2 ) MDR
  group by   SEROTYPE, year  ) MDR_by_decade

  inner join

 (select  SEROTYPE, year,      count(*) as TotalIsolates
  from 
  (SELECT distinct   id ,case when Year<2012 then 'Before2012' else 'After2012' end as year
  ,case when SEROTYPE = 'dublin' then SEROTYPE else 'Non-Dublin' end as SEROTYPE
  FROM NARM_Cattle_S_HACCP_2010_2013_View
  ) All_isloates
  group by SEROTYPE, year 
 ) All_isloates_by_decade
  
 on    MDR_by_decade.year = All_isloates_by_decade.year
 and MDR_by_decade.SEROTYPE = All_isloates_by_decade.SEROTYPE
 

 order by  MDR_by_decade.SEROTYPE,MDR_by_decade.year desc
