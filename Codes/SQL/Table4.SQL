Use SalDB

-------Compare networks before and after 2012
select distinct *
from 
(SELECT   n1.[Node]  
      ,n1.[Cluster] as Cluster_Before2012
      ,n2.[Cluster] as Cluster_After2012

      ,n1.[Deg_Cent] as Centrality_Before2012
	  ,n2.[Deg_Cent] as Centrality_After2012


  FROM (select * from [Network_View] where [Sample_Condition] like '%before%'  )n1
    left join  
	(select * from [Network_View] where [Sample_Condition] like '%after%'  ) n2
  on n1.node = n2.node 


  union all

  SELECT     n1.[Node]  
      ,n1.[Cluster] as Cluster_Before2012
      ,n2.[Cluster] as Cluster_After2012

      ,n1.[Deg_Cent] as Centrality_Before2012
	  ,n2.[Deg_Cent] as Centrality_After2012

  FROM (select * from [Network_View] where [Sample_Condition] like '%before%'  )n1
    right join  
	(select * from [Network_View] where [Sample_Condition] like '%after%'  ) n2
  on n1.node = n2.node 

  ) a


  order by Cluster_Before2012, Cluster_After2012,   node
