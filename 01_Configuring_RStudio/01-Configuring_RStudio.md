# SageMaker fundamentals for R users <br/> Part 01: Configuring RStudio


This workshop module solely focuses on how to prepare and configure the
user’s RStudio environment that should be used to connect to Amazon
SageMaker. We assume that we install all necessary prerequisites from
scratch. Just execute each code chunk in this Rnotebook as you follow
along.

Installing reticulate and miniconda
-----------------------------------

We get started by launching R/RStudio to install the `reticulate`
package and `miniconda` on your machine:

``` r
install.packages("reticulate")

library(reticulate)

install_miniconda()
```

If you are using Windows, then after installing miniconda, please make
sure to add the following entries to your system `PATH` variable,
otherwise the `reticulate::conda_install()` commands below will fail
when setting the pip = TRUE argument. It took me quite a long time to
figure this out. See also this discussion on GitHub on that topic which
solved the issue:
<a href="https://github.com/pypa/virtualenv/issues/1139" class="uri">https://github.com/pypa/virtualenv/issues/1139</a>.
(If you don’t want to edit the system `PATH` variable, please check out
the *Alternative package installation* paragraph in the next section as
an alternative approach when creating our Amazon SageMaker-specific
conda environment.)

    C:\%Miniconda3_DIR%;
    C:\%Miniconda3_DIR%\Scripts;
    C:\%Miniconda3_DIR%\Library\bin

`%Miniconda3_DIR%` should be substituted by your Miniconda installation
path.

Please, close and re-open R/RStudio after editing your system `PATH`
variable (restarting the R session within RStudio won’t be enough).

Creating an Amazon SageMaker conda environment
----------------------------------------------

Next, we will create a new conda environment which we will use
throughout this course to connect to Amazon SageMaker:

``` r
library(reticulate)

conda_create("sagemaker-r")
```

If you installed everything from scratch so far, you should now see
three listed conda environments: `r-miniconda` which came with the
miniconda installation, `r-reticluate` which is the standard environment
used by the `reticulate` package, and `sagemaker-r` which we just
created:

``` r
conda_list()

##           name
## 1  r-miniconda
## 2 r-reticulate
## 3  sagemaker-r
##                                                                             python
## 1                     C:\\Users\\alexlemm\\AppData\\Local\\r-miniconda\\python.exe
## 2 C:\\Users\\alexlemm\\AppData\\Local\\r-miniconda\\envs\\r-reticulate\\python.exe
## 3  C:\\Users\\alexlemm\\AppData\\Local\\r-miniconda\\envs\\sagemaker-r\\python.exe
```

Now, we will install the necessary Python packages in our new
environment. We only install pandas in our environment to avoid getting
a warning message every time we import the Amazon SageMaker Python
module in our R scripts. Otherwise we won’t need/use pandas.

``` r
conda_install("sagemaker-r", "boto3", pip = TRUE)
conda_install("sagemaker-r", "sagemaker", pip = TRUE)
conda_install("sagemaker-r", "pandas")
```

**Alternative package installation**: If you don’t like to edit your
system `PATH` variable in order to install Python libraries using pip
from R, *skip the last code block above* and *do the following instead*:

-   Outside of this notebook open Anaconda Prompt.
-   Activate the environment you just created via
    `conda activate sagemaker-r`.
-   Install `boto3` by executing `pip install boto3`.
-   Install `sagemaker` by executing `pip install sagemaker`.
-   Install `pandas` by executing `conda install pandas`.
-   Close Anaconda Prompt.

Configuring your AWS credentials
--------------------------------

Before you can begin using `sagemaker` and `boto3`, you need to set up
your AWS authentication credentials on your machine. Credentials for
your AWS account can be found in Identity & Access Management (IAM) when
logged in to the AWS console. You can create or use an existing AWS user
depending on your privileges. Assuming that you already have an AWS user
configured, please do the following:

-   In the AWS Console navigate to IAM and click on **Users** in the
    left sidebar.
-   Click on your user in the user table and then select the **Security
    credentials** tab.
-   Click on **Create access key** and make sure to copy both the
    **Access key ID** and the **Secret access key**.

On your machine, create a credential file at `~/.aws/credentials` and
add the following (`credentials` is the name of the file and it does not
have a file extension):

    [default]
    aws_access_key_id = [YOUR_ACCESS_KEY_ID]
    aws_secret_access_key = [YOUR_SECRET_ACCESS_KEY]

You may also want to set a default region of your choice. This can be
done in the configuration file you can create at `~/.aws/config` (Again,
`config` is the name of the file and it does not have a file extension).
We will use `us-east-1` as our standard region:

    [default]
    region=us-east-1

**Important**: Make sure to set a default region in which the SageMaker
service is available. See the [AWS region
table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/)
for the latest information on regional service availability.

Creating an IAM role for Amazon SageMaker
-----------------------------------------

Functions that we will use to execute training and tuning jobs on Amazon
SageMaker later require to pass an IAM role that will give Amazon
SageMaker permission to perform actions in other AWS services on your
behalf. For instance, the Amazon SageMaker training jobs use this role
to access training data that resides in S3. Roles are specified via
Amazon Resource Names (ARNs) that uniquely identify AWS resources.

Below we describe two alternative ways to create the role: Via the AWS
Console or programmatically.

After we will have created the role, we will copy its ARN and save it as
an R environment variable.

### Option 1: Creating the role via the AWS Console

We need to log on to the AWS console and create the respective role
while configuring the launch options of an Amazon SageMaker notebook.

To do so, please execute the following steps:

-   Log in to the AWS Console and navigate to the SageMaker service.
-   In the left sidebar go to **Notebook instances** and click on
    **Create notebook instance**.
-   Go to the **Permissions and encryption** section and select **Create
    new IAM role** from the drop-down menu.
-   In the **Create an IAM role window** under **S3 buckets you
    specify - optional** select **None** and go with the remaining
    defaults.
-   Click on **Create role**
-   You will now see a message *“Success! You created an IAM role”*.
    Click on the link that is part of the message that will bring you to
    IAM.
-   Copy the Role ARN which has the following format to your clipboard:
    `arn:aws:iam::[YOUR_ACCOUNT_ID]:role/service-role/AmazonSageMaker-ExecutionRole-[RESOURCE_ID]`.
-   You can now leave the AWS Console and you do not need to create the
    notebook instance.

### Option 2: Creating the role programmatically

You can also create the role you generated in the section above with the
exact same permissions programmatically by executing the following code:

``` r
# Activate the conda environment we created earlier
use_condaenv("sagemaker-r", required = TRUE)

# Create a boto3 client to access IAM
boto3 <- import("boto3")
iam_client <- boto3$client("iam")

time_stamp <- format(Sys.time(), "%Y%m%dT%H%M%S")

role_name = paste0("AmazonSageMaker-ExecutionRole-", time_stamp)

# A trust relationship policy defines which entity can assume this role. 
# In our case that will be the Amazon SageMaker service
trust_relationship_policy <- '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Creating the new IAM role
resp <- iam_client$create_role(Path = "/service-role/",
                        RoleName = role_name,
                        AssumeRolePolicyDocument = trust_relationship_policy,
                        Description = "SageMaker execution role created from RStudio")

# This is the role's ARN we need to save as an environment variable later! 
role_arn <- resp$Role$Arn

# Attach the AWS managed policy "AmazonSageMakerFullAccess" to the role. 
# This policy provides full access to Amazon SageMaker via the AWS Management Console and SDK. 
# It also provides selected access to related services (e.g., S3, ECR, CloudWatch Logs). 
resp <- iam_client$attach_role_policy(RoleName = role_name, 
                                      PolicyArn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")

# Create a new managed policy which also will be attached to the new role. 
policy_document <- '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "s3:ListBucket"
            ],
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::SageMaker"
            ]
        },
        {
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::SageMaker/*"
            ]
        }
    ]
}'

resp <- iam_client$create_policy(PolicyName = paste0("AmazonSageMaker-ExecutionPolicy-", time_stamp),
                                 Path = "/service-role/",
                                 PolicyDocument = policy_document)
policy_arn <- resp$Policy$Arn
resp <- iam_client$attach_role_policy(RoleName = role_name,
                                      PolicyArn = policy_arn)

# Print the role's ARN to the console and copy it to the clipboard
role_arn
```

### Saving the role ARN as an R environment variable

Now that we have the new role’s ARN, we will save it as an R environment
variable under `.Renivron`. You can easily edit `.Renviron` using
`usethis::edit_r_environ()`. Please add the following environment
variable to the file and paste the ARN from your clipboard:

    SAGEMAKER_ROLE_ARN = [YOUR_SAGEMAKER_ROLE_ARN]

Testing your setup
------------------

We are finally ready to test our setup! In RStudio select the new conda
environment we created earlier:

``` r
use_condaenv("sagemaker-r", required = TRUE)
```

Execute the following lines of code which import the SageMaker Python
module, create a SageMaker Session object and create a default Amazon S3
bucket for our sessions. Calling `default_bucket()` the first time will
create the default bucket based on the following format:
`sagemaker-[YOUR_REGION]-[YOUR_AWS_ACCOUNT_ID]`:

``` r
sagemaker <- import("sagemaker")
session <- sagemaker$Session()

my_bucket <- session$default_bucket()
my_bucket
```

If everything was configured correctly, you should have been able to
execute the four lines of code above successfully and now see the name
of your default Amazon S3 bucket printed to the console.

Summary
-------

Congratulations, you made it through the entire configuration journey!
In this module we showed how to configure your own RStudio environment
to connect to Amazon SageMaker as an alternative to Amazon SageMaker
Notebooks. In the next workshop modules the real fun part will begin and
we will use one example project to run through the entire machine
learning process using Amazon SageMaker from RStudio. We will start with
training and evaluating a XGBoost model in the next module.
