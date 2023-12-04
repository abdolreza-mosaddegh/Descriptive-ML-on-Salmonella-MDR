Use SalDB

-------Compare associations among resistances before and after 2012
select Association
, max(Support_Before2012) as Support_Before2012
, max(Support_After2012) as Support_After2012
, max(Sup_Difference) as Difference_in_Support_After2012
, max(Confidence_Before2012) as Confidence_Before2012
, max(Confidence_After2012) as Confidence_After2012
, max(Conf_Difference) as Difference_in_Confidence_After2012
, max(Lift_Before2012) as Lift_Before2012
, max(Lift_After2012) as Lift_After2012
, max(Lift_Difference) as Difference_in_Lift_After2012
 
from
(select Rule_type 
,replace(Association, ':resistant','') as Association
, Support_Before2012
, Support_After2012
, Confidence_Before2012
, Confidence_After2012
, Lift_Before2012
, Lift_After2012
, Sup_Difference,Conf_Difference,Lift_Difference
 
from
(select t.*
, case when abs(Supp_Def_Per) = 0 then ''
       when abs(Supp_Def_Per) <=25 then ''
	   else '* '
  end  + cast(cast( (Supp_Def_Per) as int) as varchar(5)) + ' %' as Sup_Difference
, case when abs(Conf_Def_Per) = 0 then ''
       when abs(Conf_Def_Per) <=10 then ''
	   else '* '
  end  + cast(cast( (Conf_Def_Per) as int) as varchar(5)) + ' %' as Conf_Difference
, case when abs(Lift_Def_Per) = 0 then ''
       when abs(Lift_Def_Per) <=25 then ''
	   else '* '
  end  + cast(cast( (Lift_Def_Per) as int) as varchar(5)) + ' %' as Lift_Difference
 
from
(
select a.ruleid,a.association

, a.real_support as Support_Before2012
, b.real_support as Support_After2012
,(b.real_support - a.real_support)*100.0/ a.real_support    as Supp_Def_Per 
 
, a.confidence as Confidence_Before2012
, b.confidence as Confidence_After2012
,(b.confidence - a.confidence)*100.0/ a.confidence    as Conf_Def_Per
 
, a.lift as Lift_Before2012
, b.lift as Lift_After2012
,(b.lift - a.lift)*100.0/  a.lift    as Lift_Def_Per 
 
 
from
(
SELECT distinct substring(a.ruleid,1,19) as group_id, a.*
  FROM [AssociationRule] a
  inner join
  Attribute b
  on a.ruleid = b.ruleid
     where a.sampling_condition like '%< 2012%'
	 and a.real_support >= 0.1
	 and a.support >= 0.05
	 and a.confidence >= 0.8
	 and a.lift >= 1.5
   ) a
   inner join
   (
SELECT distinct substring(a.ruleid,1,19) as group_id, a.*
  FROM [AssociationRule] a
  inner join
  Attribute b
  on a.ruleid = b.ruleid
     where a.sampling_condition like '%>= 2012%'
   ) b
   on a.association = b.association

   ) t
 ) kk 
 inner join
 (select ruleid , case when max(class) is null then 'Intra_Class' when max(class) = min(class) then  'Intra_Class' else 'Cross_Class' end as rule_type
   from Attribute a
   left join
   (select drug , case when class in    ('Cephalosporins','Penicillins', 'Beta lactam','Carbapenems' )  then 'Beta Lactam' else Class end as Class from drug) d
   on d.drug = a.attribute
   group by ruleid
 
 ) bb
 on kk.ruleid = bb.ruleid


 union all

 select Rule_type 
, replace(Association, ':resistant','') as Association
, Support_Before2012
, Support_After2012
, Confidence_Before2012
, Confidence_After2012
, Lift_Before2012
, Lift_After2012
, Sup_Difference,Conf_Difference,Lift_Difference
 
from
(select t.*
, case when abs(Supp_Def_Per) = 0 then ''
       when abs(Supp_Def_Per) <=25 then ''
	   else '* '
  end  + cast(cast( (Supp_Def_Per) as int) as varchar(5)) + ' %' as Sup_Difference
, case when abs(Conf_Def_Per) = 0 then ''
       when abs(Conf_Def_Per) <=10 then ''
	   else '* '
  end  + cast(cast( (Conf_Def_Per) as int) as varchar(5)) + ' %' as Conf_Difference
, case when abs(Lift_Def_Per) = 0 then ''
       when abs(Lift_Def_Per) <=25 then ''
	   else '* '
  end  + cast(cast( (Lift_Def_Per) as int) as varchar(5)) + ' %' as Lift_Difference
 
from
(
select a.ruleid,a.association

, a.real_support as Support_Before2012
, b.real_support as Support_After2012
,(b.real_support - a.real_support)*100.0/ a.real_support    as Supp_Def_Per 
 
, a.confidence as Confidence_Before2012
, b.confidence as Confidence_After2012
,(b.confidence - a.confidence)*100.0/ a.confidence    as Conf_Def_Per
 
, a.lift as Lift_Before2012
, b.lift as Lift_After2012
,(b.lift - a.lift)*100.0/  a.lift    as Lift_Def_Per 

 
from
(
SELECT distinct substring(a.ruleid,1,19) as group_id, a.*
  FROM [AssociationRule] a
  inner join
  Attribute b
  on a.ruleid = b.ruleid
     where a.sampling_condition like '%< 2012%'
	 ) a
   inner join
   (
SELECT distinct substring(a.ruleid,1,19) as group_id, a.*
  FROM [AssociationRule] a
  inner join
  Attribute b
  on a.ruleid = b.ruleid
     where a.sampling_condition like '%>= 2012%'
     and a.real_support >= 0.1
   	 and a.support >= 0.05
	 and a.confidence >= 0.8
	 and a.lift >= 1.5
   ) b
   on a.association = b.association

   ) t
 ) kk 
 inner join
 (select ruleid , case when max(class) is null then 'Intra_Class' when max(class) = min(class) then  'Intra_Class' else 'Cross_Class' end as rule_type
   from Attribute a
   left join
   (select drug , case when class in    ('Cephalosporins','Penicillins', 'Beta lactam','Carbapenems' )  then 'Beta Lactam' else Class end as Class from drug) d
   on d.drug = a.attribute
   group by ruleid
 
 ) bb
 on kk.ruleid = bb.ruleid

 ) a

 where Association like '%serotype%'
   and Sup_Difference like '%* %'
   and lift_Difference not like '%* %'
 group by  Rule_type,association



