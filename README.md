# NN Classification
 
Dataset used is a subset of LendingClub DataSet obtained from Kaggle. The subset itself was modified by the organizers of the course "Python for Data Science and Machine Learning Bootcamp" on Udemy, so that it provides possibilities for feature engineering.

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. LendingClub is the world's largest peer-to-peer lending platform.

Given historical data on loans given out with information on whether or not the borrower defaulted (charged off), the goal is to build a model that can predict whether or nor a borrower will pay back their loan. This way in the future, a new potential customer shows up, it can be assumed if he/she will pay back or not.

The "loan_status" column contains the label ("Charged Off" or "Fully Paid"). The original dataset contains:

0   loan_amnt				The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.

1	term					The number of payments on the loan. Values are in months and can be either 36 or 60.

2	int_rate				Interest Rate on the loan

3	installment				The monthly payment owed by the borrower if the loan originates.

4	grade					LC assigned loan grade

5	sub_grade				LC assigned loan subgrade

6	emp_title				The job title supplied by the Borrower when applying for the loan.

7	emp_length				Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.'\n'
8	home_ownership			The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER'\n'
9	annual_inc				The self-reported annual income provided by the borrower during registration.'\n'
10	verification_status		Indicates if income was verified by LC, not verified, or if the income source was verified'\n'
11	issue_d					The month which the loan was funded'\n'
12	loan_status				Current status of the loan'\n'
13	purpose					A category provided by the borrower for the loan request.'\n'
14	title					The loan title provided by the borrower'\n'
15	zip_code				The first 3 numbers of the zip code provided by the borrower in the loan application.'\n'
16	addr_state				The state provided by the borrower in the loan application'\n'
17	dti						A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.'\n'
18	earliest_cr_line		The month the borrower's earliest reported credit line was opened'\n'
19	open_acc				         The number of open credit lines in the borrower's credit file.'\n'
20	pub_rec					         Number of derogatory public records'\n'
21	revol_bal				        Total credit revolving balance'\n'
22	revol_util				       Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.'\n'
23	total_acc				        The total number of credit lines currently in the borrower's credit file'\n'
24	initial_list_status		The initial listing status of the loan. Possible values are – W, F'\n'
25	application_type		   Indicates whether the loan is an individual application or a joint application with two co-borrowers'\n'
26	mort_acc				         Number of mortgage accounts.'\n'
27	pub_rec_bankruptcies	Number of public record bankruptcies'\n'
