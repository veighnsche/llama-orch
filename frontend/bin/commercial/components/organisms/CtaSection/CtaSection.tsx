import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, BookOpen, MessageCircle } from "lucide-react"
import { SectionContainer } from '@/components/molecules'

export function CTASection() {
  return (
    <SectionContainer 
      title={
        <>
          Take Control of Your<br />
          <span className="text-primary">AI Infrastructure Today.</span>
        </>
      }
      maxWidth="4xl"
    >

      <p className="text-xl text-muted-foreground leading-relaxed text-center">
        Join hundreds of users, providers, and enterprises who've chosen independence.
      </p>

      <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
        <Button
          size="lg"
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"
        >
          Get Started Free
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
        >
          <BookOpen className="mr-2 h-5 w-5" />
          View Documentation
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
        >
          <MessageCircle className="mr-2 h-5 w-5" />
          Join Discord
        </Button>
      </div>

      <p className="text-muted-foreground pt-4 text-center">100% open source. No credit card required. Install in 15 minutes.</p>
    </SectionContainer>
  )
}
