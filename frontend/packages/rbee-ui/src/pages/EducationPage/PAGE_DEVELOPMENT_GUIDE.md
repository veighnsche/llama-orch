# EducationPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/education` (Education & Learning)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Education page showcasing rbee for teaching distributed AI systems and hands-on learning.

**Target Audience:** Educators, students, bootcamps, universities, online course creators

**Key Message:** Teach distributed AI with real infrastructure. Students learn by doing, not just watching.

---

## 🔄 Key Template Adaptations for Education

### Hero
- ✅ `HeroTemplate` with learning path visualization

### Problem: Theoretical Learning
- ✅ `ProblemTemplate` - No hands-on experience, expensive cloud labs, limited access

### Solution: Real Infrastructure
- ✅ `SolutionTemplate` - Hands-on learning, real orchestration, production patterns

### Learning Paths
- ✅ `PricingTemplate` adapted - Course levels (Beginner → Intermediate → Advanced)
- ✅ `ComparisonTemplate` adapted - Learning path comparison

### Curriculum Modules
- ✅ `EnterpriseSecurity` adapted - 6 module cards (Basics, Orchestration, Multi-GPU, etc.)
- ✅ `FeaturesTabs` adapted - Module tabs with content

### Student Outcomes
- ✅ `TestimonialsTemplate` adapted - Student success stories
- ✅ Stats: Students taught, projects built, job placements

### Lab Exercises
- ✅ `HowItWorks` - Step-by-step lab exercises
- ✅ `CodeExamplesTemplate` - Code examples for labs

### Use Cases
- ✅ `UseCasesTemplate` - Different student types (beginner, career switcher, researcher)

### Time Estimator
- ✅ `ProvidersEarnings` adapted - Learning time calculator
  - Input: Current level, target level, hours/week
  - Output: Time to completion, modules to complete

---

## 📐 Proposed Structure

```tsx
<EducationPage>
  <HeroTemplate /> {/* Learning path viz */}
  <EmailCapture /> {/* "Access Course Materials" */}
  <ProblemTemplate /> {/* Theoretical learning gap */}
  <SolutionTemplate /> {/* Hands-on with real infra */}
  <PricingTemplate /> {/* Course levels */}
  <EnterpriseSecurity /> {/* Curriculum modules */}
  <HowItWorks /> {/* Lab exercises */}
  <ProvidersEarnings /> {/* Time estimator */}
  <UseCasesTemplate /> {/* Student types */}
  <TestimonialsTemplate /> {/* Student outcomes */}
  <FAQTemplate /> {/* Course FAQs */}
  <CTATemplate /> {/* "Start Learning" */}
</EducationPage>
```

---

## 📚 Learning Time Calculator

**Adapt `ProvidersEarnings`:**

```tsx
calculatorProps = {
  title: "Estimate Your Learning Path",
  inputs: [
    { type: "select", label: "Current Level", options: ["Beginner", "Intermediate", "Advanced"] },
    { type: "select", label: "Target Role", options: ["AI Engineer", "MLOps", "Research"] },
    { type: "slider", label: "Hours/Week", min: 5, max: 40 }
  ],
  output: {
    weeks: "X weeks to completion",
    modules: "Y modules to complete",
    projects: "Z hands-on projects"
  }
}
```

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `EducationPageProps.tsx`
- [ ] Adapt templates for education context
- [ ] Write curriculum-focused copy
- [ ] Create `EducationPage.tsx`
- [ ] Test and document

---

**Key Message:** Learning by doing. Show students they'll work with real distributed systems, not toy examples.
